//! WinFsp drive-letter mount (Windows, `winfsp` feature).
//!
//! Exposes a NeuralFS volume as a real Windows drive letter through the WinFsp
//! kernel-mode filesystem driver. This is a self-contained in-memory filesystem
//! that proves the kernel-driver integration end to end; the CoW `neuralfs-fs`
//! engine is the documented next backing step.
//!
//! Modelled on the winfsp-rs `memfs` example.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use winfsp::filesystem::{
    DirBuffer, DirInfo, DirMarker, FileInfo, FileSecurity, FileSystemContext, OpenFileInfo,
    VolumeInfo, WideNameInfo,
};
use winfsp::host::{DebugMode, FileSystemHost, FileSystemParams, VolumeParams};
use winfsp::service::FileSystemServiceBuilder;
use winfsp::{winfsp_init_or_die, FspError, U16CStr};

use windows::core::PCWSTR;
use windows::Win32::Foundation::{
    STATUS_ACCESS_DENIED, STATUS_END_OF_FILE, STATUS_NONCONTINUABLE_EXCEPTION,
    STATUS_OBJECT_NAME_COLLISION, STATUS_OBJECT_NAME_NOT_FOUND,
};
use windows::Win32::Security::Authorization::{
    ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
};
use windows::Win32::Security::{GetSecurityDescriptorLength, PSECURITY_DESCRIPTOR};

const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
const FILE_ATTRIBUTE_ARCHIVE: u32 = 0x20;
const FILE_DIRECTORY_FILE: u32 = 0x0000_0001;
/// Everyone / SYSTEM / Administrators: full access. Keeps the demo permissive.
const ROOT_SDDL: &str = "O:BAG:BAD:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;WD)";

/// Build the WinFsp init token; used by `--winfsp-probe`.
pub fn probe() -> anyhow::Result<()> {
    let _init = winfsp::winfsp_init().map_err(|e| anyhow::anyhow!("winfsp init failed: {e:?}"))?;
    println!("winfsp initialized OK (driver reachable, library linked)");
    Ok(())
}

fn now_filetime() -> u64 {
    let nanos_100 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() / 100)
        .unwrap_or(0) as u64;
    // FILETIME epoch (1601) is 11644473600 seconds before the Unix epoch.
    nanos_100 + 116_444_736_000_000_000u64
}

/// Parse an SDDL string into raw self-relative security-descriptor bytes.
fn sddl_bytes(sddl: &str) -> anyhow::Result<Vec<u8>> {
    let wide: Vec<u16> = sddl.encode_utf16().chain(std::iter::once(0)).collect();
    let mut descriptor = PSECURITY_DESCRIPTOR::default();
    unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            PCWSTR(wide.as_ptr()),
            SDDL_REVISION_1,
            &mut descriptor,
            None,
        )?;
    }
    if descriptor.0.is_null() {
        anyhow::bail!("SDDL parsing returned a null descriptor");
    }
    let len = unsafe { GetSecurityDescriptorLength(descriptor) } as usize;
    let bytes = unsafe { std::slice::from_raw_parts(descriptor.0 as *const u8, len) }.to_vec();
    // The descriptor is a one-time, process-lifetime allocation (~80 bytes);
    // we copy it out and intentionally don't LocalFree to avoid a windows-crate
    // version-specific import. Negligible, bounded leak.
    Ok(bytes)
}

struct Node {
    is_dir: bool,
    attributes: u32,
    data: Vec<u8>,
    creation_time: u64,
    last_access_time: u64,
    last_write_time: u64,
    change_time: u64,
    index_number: u64,
}

impl Node {
    fn file_info(&self) -> FileInfo {
        let mut fi = FileInfo::default();
        fi.file_attributes = self.attributes;
        fi.allocation_size = (self.data.len() as u64 + 4095) / 4096 * 4096;
        fi.file_size = self.data.len() as u64;
        fi.creation_time = self.creation_time;
        fi.last_access_time = self.last_access_time;
        fi.last_write_time = self.last_write_time;
        fi.change_time = self.change_time;
        fi.index_number = self.index_number;
        fi
    }
}

struct State {
    nodes: HashMap<String, Node>,
    security: Vec<u8>,
    next_index: u64,
}

/// Per-open handle. Holds the path key and (for directories) a WinFsp
/// directory buffer used to enumerate entries.
pub struct NeuralHandle {
    key: String,
    dir_buffer: DirBuffer,
}

// DirBuffer wraps a WinFsp-owned pointer; the host serializes access and the
// crate's own contexts (e.g. ntptfs) treat it as Send/Sync.
unsafe impl Send for NeuralHandle {}
unsafe impl Sync for NeuralHandle {}

/// The NeuralFS in-memory filesystem exposed over WinFsp.
pub struct NeuralWinFsContext {
    state: Mutex<State>,
}

impl NeuralWinFsContext {
    fn new() -> anyhow::Result<Self> {
        let security = sddl_bytes(ROOT_SDDL)?;
        let now = now_filetime();
        let mut nodes = HashMap::new();
        nodes.insert(
            "\\".to_string(),
            Node {
                is_dir: true,
                attributes: FILE_ATTRIBUTE_DIRECTORY,
                data: Vec::new(),
                creation_time: now,
                last_access_time: now,
                last_write_time: now,
                change_time: now,
                index_number: 1,
            },
        );
        Ok(Self {
            state: Mutex::new(State {
                nodes,
                security,
                next_index: 2,
            }),
        })
    }
}

/// Normalize a WinFsp path (`\`, `\a`, `\a\b`) to a canonical map key.
fn key_of(file_name: &U16CStr) -> String {
    let s = String::from_utf16_lossy(file_name.as_slice());
    if s.is_empty() {
        "\\".to_string()
    } else {
        s
    }
}

fn parent_of(key: &str) -> &str {
    if key == "\\" {
        return "\\";
    }
    match key.rfind('\\') {
        Some(0) => "\\",
        Some(i) => &key[..i],
        None => "\\",
    }
}

fn basename_of(key: &str) -> &str {
    match key.rfind('\\') {
        Some(i) => &key[i + 1..],
        None => key,
    }
}

fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

impl FileSystemContext for NeuralWinFsContext {
    type FileContext = NeuralHandle;

    fn get_security_by_name(
        &self,
        file_name: &U16CStr,
        security_descriptor: Option<&mut [std::ffi::c_void]>,
        _resolve_reparse: impl FnOnce(&U16CStr) -> Option<FileSecurity>,
    ) -> winfsp::Result<FileSecurity> {
        let key = key_of(file_name);
        let state = self.state.lock();
        let node = state
            .nodes
            .get(&key)
            .ok_or(FspError::NTSTATUS(STATUS_OBJECT_NAME_NOT_FOUND.0))?;
        let attributes = node.attributes;
        let sd = &state.security;
        let sd_len = sd.len() as u64;
        if let Some(buffer) = security_descriptor {
            if (buffer.len() as u64) >= sd_len {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        sd.as_ptr(),
                        buffer.as_mut_ptr() as *mut u8,
                        sd.len(),
                    );
                }
            }
        }
        Ok(FileSecurity {
            reparse: false,
            sz_security_descriptor: sd_len,
            attributes,
        })
    }

    fn open(
        &self,
        file_name: &U16CStr,
        _create_options: u32,
        _granted_access: u32,
        file_info: &mut OpenFileInfo,
    ) -> winfsp::Result<Self::FileContext> {
        let key = key_of(file_name);
        let state = self.state.lock();
        let node = state
            .nodes
            .get(&key)
            .ok_or(FspError::NTSTATUS(STATUS_OBJECT_NAME_NOT_FOUND.0))?;
        *file_info.as_mut() = node.file_info();
        Ok(NeuralHandle {
            key,
            dir_buffer: DirBuffer::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create(
        &self,
        file_name: &U16CStr,
        create_options: u32,
        _granted_access: u32,
        file_attributes: u32,
        _security_descriptor: Option<&[std::ffi::c_void]>,
        _allocation_size: u64,
        _extra_buffer: Option<&[u8]>,
        _extra_is_reparse: bool,
        file_info: &mut OpenFileInfo,
    ) -> winfsp::Result<Self::FileContext> {
        let key = key_of(file_name);
        let is_dir = (create_options & FILE_DIRECTORY_FILE) != 0;
        let mut state = self.state.lock();
        if state.nodes.contains_key(&key) {
            return Err(FspError::NTSTATUS(STATUS_OBJECT_NAME_COLLISION.0));
        }
        // Parent must exist and be a directory.
        let parent = parent_of(&key).to_string();
        match state.nodes.get(&parent) {
            Some(p) if p.is_dir => {}
            _ => return Err(FspError::NTSTATUS(STATUS_OBJECT_NAME_NOT_FOUND.0)),
        }
        let now = now_filetime();
        let index = state.next_index;
        state.next_index += 1;
        let attributes = if is_dir {
            FILE_ATTRIBUTE_DIRECTORY
        } else {
            (file_attributes & !FILE_ATTRIBUTE_DIRECTORY) | FILE_ATTRIBUTE_ARCHIVE
        };
        let node = Node {
            is_dir,
            attributes,
            data: Vec::new(),
            creation_time: now,
            last_access_time: now,
            last_write_time: now,
            change_time: now,
            index_number: index,
        };
        *file_info.as_mut() = node.file_info();
        state.nodes.insert(key.clone(), node);
        Ok(NeuralHandle {
            key,
            dir_buffer: DirBuffer::new(),
        })
    }

    fn close(&self, _context: Self::FileContext) {}

    fn cleanup(&self, context: &Self::FileContext, _file_name: Option<&U16CStr>, flags: u32) {
        use winfsp::constants::FspCleanupFlags::FspCleanupDelete;
        if FspCleanupDelete.is_flagged(flags) {
            let key = &context.key;
            let mut state = self.state.lock();
            // Refuse to delete a non-empty directory.
            let has_child = state
                .nodes
                .keys()
                .any(|k| k != key && parent_of(k) == key.as_str());
            if !has_child {
                state.nodes.remove(key);
            }
        }
    }

    fn read(
        &self,
        context: &Self::FileContext,
        buffer: &mut [u8],
        offset: u64,
    ) -> winfsp::Result<u32> {
        let state = self.state.lock();
        let node = state
            .nodes
            .get(&context.key)
            .ok_or(FspError::NTSTATUS(STATUS_OBJECT_NAME_NOT_FOUND.0))?;
        if offset >= node.data.len() as u64 {
            return Err(FspError::NTSTATUS(STATUS_END_OF_FILE.0));
        }
        let end = (offset + buffer.len() as u64).min(node.data.len() as u64);
        let len = (end - offset) as usize;
        buffer[..len].copy_from_slice(&node.data[offset as usize..offset as usize + len]);
        Ok(len as u32)
    }

    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        context: &Self::FileContext,
        buffer: &[u8],
        offset: u64,
        write_to_eof: bool,
        constrained_io: bool,
        file_info: &mut FileInfo,
    ) -> winfsp::Result<u32> {
        let mut state = self.state.lock();
        let node = state
            .nodes
            .get_mut(&context.key)
            .ok_or(FspError::NTSTATUS(STATUS_ACCESS_DENIED.0))?;
        let start = if write_to_eof {
            node.data.len() as u64
        } else {
            offset
        };
        if constrained_io && start >= node.data.len() as u64 {
            *file_info = node.file_info();
            return Ok(0);
        }
        let end = if constrained_io {
            (start + buffer.len() as u64).min(node.data.len() as u64)
        } else {
            start + buffer.len() as u64
        };
        if end as usize > node.data.len() {
            node.data.resize(end as usize, 0);
        }
        let len = (end - start) as usize;
        node.data[start as usize..start as usize + len].copy_from_slice(&buffer[..len]);
        node.last_write_time = now_filetime();
        *file_info = node.file_info();
        Ok(len as u32)
    }

    fn get_file_info(
        &self,
        context: &Self::FileContext,
        file_info: &mut FileInfo,
    ) -> winfsp::Result<()> {
        let state = self.state.lock();
        let node = state
            .nodes
            .get(&context.key)
            .ok_or(FspError::NTSTATUS(STATUS_OBJECT_NAME_NOT_FOUND.0))?;
        *file_info = node.file_info();
        Ok(())
    }

    fn set_file_size(
        &self,
        context: &Self::FileContext,
        new_size: u64,
        _set_allocation_size: bool,
        file_info: &mut FileInfo,
    ) -> winfsp::Result<()> {
        let mut state = self.state.lock();
        let node = state
            .nodes
            .get_mut(&context.key)
            .ok_or(FspError::NTSTATUS(STATUS_OBJECT_NAME_NOT_FOUND.0))?;
        node.data.resize(new_size as usize, 0);
        node.last_write_time = now_filetime();
        *file_info = node.file_info();
        Ok(())
    }

    fn set_basic_info(
        &self,
        context: &Self::FileContext,
        file_attributes: u32,
        creation_time: u64,
        last_access_time: u64,
        last_write_time: u64,
        change_time: u64,
        file_info: &mut FileInfo,
    ) -> winfsp::Result<()> {
        let mut state = self.state.lock();
        let node = state
            .nodes
            .get_mut(&context.key)
            .ok_or(FspError::NTSTATUS(STATUS_OBJECT_NAME_NOT_FOUND.0))?;
        if file_attributes != u32::MAX && file_attributes != 0 {
            node.attributes = file_attributes;
        }
        if creation_time != 0 {
            node.creation_time = creation_time;
        }
        if last_access_time != 0 {
            node.last_access_time = last_access_time;
        }
        if last_write_time != 0 {
            node.last_write_time = last_write_time;
        }
        if change_time != 0 {
            node.change_time = change_time;
        }
        *file_info = node.file_info();
        Ok(())
    }

    fn rename(
        &self,
        _context: &Self::FileContext,
        file_name: &U16CStr,
        new_file_name: &U16CStr,
        replace_if_exists: bool,
    ) -> winfsp::Result<()> {
        let from = key_of(file_name);
        let to = key_of(new_file_name);
        let mut state = self.state.lock();
        if state.nodes.contains_key(&to) && !replace_if_exists {
            return Err(FspError::NTSTATUS(STATUS_OBJECT_NAME_COLLISION.0));
        }
        // Move the node itself plus any descendants (prefix rename).
        let affected: Vec<String> = state
            .nodes
            .keys()
            .filter(|k| *k == &from || k.starts_with(&format!("{from}\\")))
            .cloned()
            .collect();
        for k in affected {
            let node = state.nodes.remove(&k).unwrap();
            let new_key = if k == from {
                to.clone()
            } else {
                format!("{to}{}", &k[from.len()..])
            };
            state.nodes.insert(new_key, node);
        }
        Ok(())
    }

    fn read_directory(
        &self,
        context: &Self::FileContext,
        _pattern: Option<&U16CStr>,
        marker: DirMarker,
        buffer: &mut [u8],
    ) -> winfsp::Result<u32> {
        let dir = context.key.clone();

        // Fill the WinFsp directory buffer once (when reset is requested for
        // this enumeration); subsequent reads stream from it via the marker.
        if let Ok(lock) = context.dir_buffer.acquire(marker.is_none(), None) {
            let state = self.state.lock();
            let mut entries: Vec<(String, FileInfo)> = Vec::new();
            if let Some(node) = state.nodes.get(&dir) {
                entries.push((".".to_string(), node.file_info()));
            }
            if let Some(pnode) = state.nodes.get(parent_of(&dir)) {
                entries.push(("..".to_string(), pnode.file_info()));
            }
            let mut children: Vec<(String, FileInfo)> = state
                .nodes
                .iter()
                .filter(|(k, _)| k.as_str() != dir.as_str() && parent_of(k) == dir.as_str())
                .map(|(k, n)| (basename_of(k).to_string(), n.file_info()))
                .collect();
            children.sort_by(|a, b| a.0.cmp(&b.0));
            entries.extend(children);

            for (name, fi) in entries {
                let mut dir_info: DirInfo<255> = DirInfo::new();
                *dir_info.file_info_mut() = fi;
                dir_info.set_name_raw(utf16(&name).as_slice())?;
                lock.write(&mut dir_info)?;
            }
        }
        Ok(context.dir_buffer.read(marker, buffer))
    }

    fn get_volume_info(&self, out_volume_info: &mut VolumeInfo) -> winfsp::Result<()> {
        let state = self.state.lock();
        let used: u64 = state.nodes.values().map(|n| n.data.len() as u64).sum();
        let total = 1u64 << 32; // 4 GiB nominal
        out_volume_info.total_size = total;
        out_volume_info.free_size = total.saturating_sub(used);
        out_volume_info.set_volume_label(std::ffi::OsStr::new("NeuralFS"));
        Ok(())
    }
}

/// Owns the running host so the service can stop it on teardown.
pub struct NeuralWinFs {
    host: FileSystemHost<NeuralWinFsContext>,
}

fn build_and_mount(drive: &str) -> anyhow::Result<NeuralWinFs> {
    let mut volume_params = VolumeParams::new();
    // Metadata cache timeouts (milliseconds). WinFsp caches these results
    // kernel-side, so within the window a query is answered without a round
    // trip down to this user-mode filesystem — and on Windows that round trip
    // is two process context switches, the documented bottleneck for repeated
    // opens and stats. This volume is fully authoritative and is only ever
    // mutated *through* WinFsp, so caching harder cannot serve stale data from
    // an out-of-band writer; we therefore raise these well above the original
    // 1 s. file_info_timeout is the base; the per-class timeouts below override
    // it for their category. The security descriptor is a fixed constant for
    // the life of the process, so it can be cached far longer than file info.
    volume_params
        .sector_size(512)
        .sectors_per_allocation_unit(1)
        .volume_creation_time(now_filetime())
        .file_info_timeout(10_000)
        .dir_info_timeout(10_000)
        .volume_info_timeout(10_000)
        .security_timeout(60_000)
        .case_sensitive_search(false)
        .case_preserved_names(true)
        .unicode_on_disk(true)
        .persistent_acls(true)
        .post_cleanup_when_modified_only(true)
        .filesystem_name("NeuralFS");

    let context = NeuralWinFsContext::new()?;
    let fs_params = FileSystemParams {
        use_dir_info_by_name: false,
        volume_params,
        debug_mode: DebugMode::none(),
    };
    let mut host = FileSystemHost::<NeuralWinFsContext>::new_with_options(fs_params, context)?;
    host.mount(drive)?;
    host.start()?;
    Ok(NeuralWinFs { host })
}

/// Mount the NeuralFS WinFsp filesystem at `drive` (e.g. "N:") and run until
/// the service is stopped (Ctrl-C / unmount / process termination).
pub fn run(drive: &str) -> anyhow::Result<()> {
    let init = winfsp_init_or_die();
    let drive = drive.to_string();
    let mut service = FileSystemServiceBuilder::new()
        .with_start(move || {
            build_and_mount(&drive).map_err(|e| {
                eprintln!("mount failed: {e:#}");
                FspError::NTSTATUS(STATUS_NONCONTINUABLE_EXCEPTION.0)
            })
        })
        .with_stop(|fs: Option<&mut NeuralWinFs>| {
            if let Some(fs) = fs {
                fs.host.stop();
            }
            Ok(())
        })
        .build("neuralfs-winfsp", init)
        .map_err(|e| anyhow::anyhow!("failed to build winfsp service: {e:?}"))?;

    service
        .start()
        .map_err(|e| anyhow::anyhow!("failed to start winfsp service: {e:?}"))?;
    let _ = service.join();
    Ok(())
}
