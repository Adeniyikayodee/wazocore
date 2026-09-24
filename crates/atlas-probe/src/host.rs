//! Collects host facts from the OS. Every probe is best-effort: a fact the
//! platform hides (Android sandboxing, macOS) becomes `None`, never an error.

#[cfg(any(target_os = "linux", target_os = "android"))]
use std::fs;
use std::process::Command;

#[cfg(any(target_os = "linux", target_os = "android"))]
use crate::cpuinfo::{self, CoreInfo};
use crate::manifest::{BlockDevice, Cluster, Host, Thermal};

pub fn collect() -> Host {
    let logical_cpus = thread_count();
    let platform = platform(logical_cpus);
    Host {
        hostname: command("hostname", &[]),
        device_model: platform.device_model,
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        kernel: command("uname", &["-r"]),
        cpu_model: platform.cpu_model,
        logical_cpus,
        clusters: platform.clusters,
        isa_features: isa_features(),
        ram_total_bytes: platform.ram_total_bytes,
        ram_available_bytes: platform.ram_available_bytes,
        cpu_governor: platform.cpu_governor,
        thermal_c: platform.thermal_c,
        block_devices: platform.block_devices,
        git_commit: command("git", &["rev-parse", "--short", "HEAD"]),
    }
}

/// The platform-specific subset of `Host`.
#[derive(Default)]
struct Platform {
    device_model: Option<String>,
    cpu_model: Option<String>,
    clusters: Vec<Cluster>,
    ram_total_bytes: Option<u64>,
    ram_available_bytes: Option<u64>,
    cpu_governor: Option<String>,
    thermal_c: Vec<Thermal>,
    block_devices: Vec<BlockDevice>,
}

fn thread_count() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

fn command(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let text = text.trim();
    (out.status.success() && !text.is_empty()).then(|| text.to_owned())
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn read_trimmed(path: &str) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    let text = text.trim_matches(|c: char| c.is_whitespace() || c == '\0');
    (!text.is_empty()).then(|| text.to_owned())
}

fn isa_features() -> Vec<&'static str> {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::is_aarch64_feature_detected as has;
        [
            ("neon", has!("neon")),
            ("dotprod", has!("dotprod")),
            ("i8mm", has!("i8mm")),
            ("fp16", has!("fp16")),
            ("sve", has!("sve")),
        ]
        .into_iter()
        .filter_map(|(name, on)| on.then_some(name))
        .collect()
    }
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        [
            ("avx2", has!("avx2")),
            ("fma", has!("fma")),
            ("f16c", has!("f16c")),
            ("avx512f", has!("avx512f")),
        ]
        .into_iter()
        .filter_map(|(name, on)| on.then_some(name))
        .collect()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        Vec::new()
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn platform(_logical_cpus: usize) -> Platform {
    let info = fs::read_to_string("/proc/cpuinfo")
        .map(|t| cpuinfo::parse_proc_cpuinfo(&t))
        .unwrap_or_default();
    let cores: Vec<CoreInfo> = info
        .cpus
        .iter()
        .map(|e| CoreInfo {
            cpu: e.index,
            core: cpuinfo::core_label(e),
            max_khz: read_trimmed(&format!(
                "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq",
                e.index
            ))
            .and_then(|s| s.parse().ok()),
        })
        .collect();
    let clusters = cpuinfo::clusters(&cores);
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    Platform {
        device_model: android_model()
            .or_else(|| read_trimmed("/proc/device-tree/model"))
            .or(info.machine),
        cpu_model: info.cpus.first().and_then(|e| e.model_name.clone()).or_else(|| {
            let names: Vec<String> =
                clusters.iter().map(|c| format!("{}x {}", c.count, c.core)).collect();
            (!names.is_empty()).then(|| names.join(" + "))
        }),
        clusters,
        ram_total_bytes: cpuinfo::meminfo_bytes(&meminfo, "MemTotal"),
        ram_available_bytes: cpuinfo::meminfo_bytes(&meminfo, "MemAvailable"),
        cpu_governor: read_trimmed("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        thermal_c: thermal_zones(),
        block_devices: block_devices(),
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn android_model() -> Option<String> {
    let maker = command("getprop", &["ro.product.manufacturer"])?;
    let model = command("getprop", &["ro.product.model"])?;
    let soc = command("getprop", &["ro.soc.model"]);
    Some(match soc {
        Some(soc) => format!("{maker} {model} ({soc})"),
        None => format!("{maker} {model}"),
    })
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn dir_names(path: &str) -> Vec<String> {
    let names: Vec<String> = fs::read_dir(path)
        .map(|rd| rd.filter_map(|e| e.ok()?.file_name().into_string().ok()).collect())
        .unwrap_or_default();
    let mut sorted = names;
    sorted.sort();
    sorted
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn thermal_zones() -> Vec<Thermal> {
    dir_names("/sys/class/thermal")
        .into_iter()
        .filter(|n| n.starts_with("thermal_zone"))
        .filter_map(|n| {
            let base = format!("/sys/class/thermal/{n}");
            let milli: f64 = read_trimmed(&format!("{base}/temp"))?.parse().ok()?;
            let zone = read_trimmed(&format!("{base}/type")).unwrap_or(n);
            Some(Thermal { zone, celsius: milli / 1000.0 })
        })
        .collect()
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn block_devices() -> Vec<BlockDevice> {
    dir_names("/sys/block")
        .into_iter()
        .filter_map(|name| {
            let base = format!("/sys/block/{name}/device");
            let mmc_type = read_trimmed(&format!("{base}/type"));
            let kind = cpuinfo::block_kind(&name, mmc_type.as_deref())?;
            let model = read_trimmed(&format!("{base}/model"))
                .or_else(|| read_trimmed(&format!("{base}/name")));
            Some(BlockDevice { name, kind, model })
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn sysctl(key: &str) -> Option<String> {
    command("sysctl", &["-n", key])
}

#[cfg(target_os = "macos")]
fn platform(_logical_cpus: usize) -> Platform {
    let levels: usize = sysctl("hw.nperflevels").and_then(|s| s.parse().ok()).unwrap_or(0);
    let clusters = (0..levels)
        .filter_map(|l| {
            let count = sysctl(&format!("hw.perflevel{l}.logicalcpu"))?.parse().ok()?;
            let core = sysctl(&format!("hw.perflevel{l}.name")).unwrap_or_else(|| format!("level{l}"));
            Some(Cluster { core, max_khz: None, count, cpus: Vec::new() })
        })
        .collect();
    Platform {
        device_model: sysctl("hw.model"),
        cpu_model: sysctl("machdep.cpu.brand_string"),
        clusters,
        ram_total_bytes: sysctl("hw.memsize").and_then(|s| s.parse().ok()),
        ..Platform::default()
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
fn platform(_logical_cpus: usize) -> Platform {
    Platform::default()
}
