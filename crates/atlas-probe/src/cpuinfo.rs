//! Pure parsers for Linux/Android `/proc` and `/sys` text, and CPU naming.

use crate::manifest::Cluster;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CpuEntry {
    pub index: usize,
    pub implementer: Option<u32>,
    pub part: Option<u32>,
    pub model_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CpuInfo {
    pub cpus: Vec<CpuEntry>,
    /// Board-level `Model` / `Hardware` line (e.g. "Raspberry Pi 5 Model B").
    pub machine: Option<String>,
}

/// Per-CPU facts gathered by the shell, grouped into clusters by `clusters`.
#[derive(Debug, Clone, PartialEq)]
pub struct CoreInfo {
    pub cpu: usize,
    pub core: String,
    pub max_khz: Option<u64>,
}

fn field(line: &str) -> Option<(&str, &str)> {
    line.split_once(':').map(|(k, v)| (k.trim(), v.trim()))
}

fn parse_hex(v: &str) -> Option<u32> {
    u32::from_str_radix(v.trim_start_matches("0x"), 16).ok()
}

fn parse_block(block: &str) -> Option<CpuEntry> {
    let fields: Vec<(&str, &str)> = block.lines().filter_map(field).collect();
    let get = |key: &str| fields.iter().find(|(k, _)| *k == key).map(|(_, v)| *v);
    Some(CpuEntry {
        index: get("processor")?.parse().ok()?,
        implementer: get("CPU implementer").and_then(parse_hex),
        part: get("CPU part").and_then(parse_hex),
        model_name: get("model name").map(str::to_owned),
    })
}

pub fn parse_proc_cpuinfo(text: &str) -> CpuInfo {
    let machine = text
        .lines()
        .filter_map(field)
        .find(|(k, _)| *k == "Model" || *k == "Hardware")
        .map(|(_, v)| v.to_owned());
    CpuInfo { cpus: text.split("\n\n").filter_map(parse_block).collect(), machine }
}

/// Arm Ltd. (implementer 0x41) core names by MIDR part number.
pub fn arm_core_name(implementer: u32, part: u32) -> Option<&'static str> {
    if implementer != 0x41 {
        return None;
    }
    Some(match part {
        0xd03 => "Cortex-A53",
        0xd04 => "Cortex-A35",
        0xd05 => "Cortex-A55",
        0xd07 => "Cortex-A57",
        0xd08 => "Cortex-A72",
        0xd09 => "Cortex-A73",
        0xd0a => "Cortex-A75",
        0xd0b => "Cortex-A76",
        0xd0d => "Cortex-A77",
        0xd41 => "Cortex-A78",
        0xd44 => "Cortex-X1",
        0xd46 => "Cortex-A510",
        0xd47 => "Cortex-A710",
        0xd48 => "Cortex-X2",
        0xd4d => "Cortex-A715",
        0xd4e => "Cortex-X3",
        0xd80 => "Cortex-A520",
        0xd81 => "Cortex-A720",
        0xd82 => "Cortex-X4",
        _ => return None,
    })
}

pub fn core_label(e: &CpuEntry) -> String {
    match (e.implementer, e.part, &e.model_name) {
        (Some(i), Some(p), _) => arm_core_name(i, p)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("arm-{i:#x}-{p:#x}")),
        (_, _, Some(name)) => name.clone(),
        _ => "unknown".to_owned(),
    }
}

/// Groups CPUs by (core type, max clock), in first-seen order.
pub fn clusters(cores: &[CoreInfo]) -> Vec<Cluster> {
    let mut out: Vec<Cluster> = Vec::new();
    for c in cores {
        match out.iter_mut().find(|k| k.core == c.core && k.max_khz == c.max_khz) {
            Some(k) => {
                k.count += 1;
                k.cpus.push(c.cpu);
            }
            None => out.push(Cluster {
                core: c.core.clone(),
                max_khz: c.max_khz,
                count: 1,
                cpus: vec![c.cpu],
            }),
        }
    }
    out
}

/// A `/proc/meminfo` entry in bytes (the file reports kB).
pub fn meminfo_bytes(text: &str, key: &str) -> Option<u64> {
    let (_, v) = text.lines().filter_map(field).find(|(k, _)| *k == key)?;
    let kib: u64 = v.split_whitespace().next()?.parse().ok()?;
    Some(kib * 1024)
}

/// Storage class from a `/sys/block` name, refined by the MMC `device/type`
/// ("MMC" = eMMC, "SD" = SD card) where present.
pub fn block_kind(name: &str, mmc_type: Option<&str>) -> Option<&'static str> {
    let virtual_dev = ["loop", "ram", "zram", "dm-", "md", "sr"];
    if virtual_dev.iter().any(|p| name.starts_with(p)) {
        return None;
    }
    Some(if name.starts_with("mmcblk") {
        match mmc_type {
            Some("MMC") => "emmc",
            Some("SD") => "sd",
            _ => "mmc",
        }
    } else if name.starts_with("nvme") {
        "nvme"
    } else if name.starts_with("sd") {
        // UFS on Android; USB or SATA elsewhere.
        "scsi"
    } else {
        "other"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI5: &str = "processor\t: 0\nBogoMIPS\t: 108.00\nFeatures\t: fp asimd asimddp\nCPU implementer\t: 0x41\nCPU part\t: 0xd0b\n\nprocessor\t: 1\nCPU implementer\t: 0x41\nCPU part\t: 0xd0b\n\nRevision\t: d04170\nModel\t\t: Raspberry Pi 5 Model B Rev 1.0\n";

    #[test]
    fn parses_pi5_cpuinfo() {
        let info = parse_proc_cpuinfo(PI5);
        assert_eq!(info.cpus.len(), 2);
        assert_eq!(core_label(&info.cpus[1]), "Cortex-A76");
        assert_eq!(info.machine.as_deref(), Some("Raspberry Pi 5 Model B Rev 1.0"));
    }

    #[test]
    fn parses_x86_model_name() {
        let info = parse_proc_cpuinfo("processor\t: 0\nmodel name\t: AMD Ryzen 7\nflags\t: avx2\n");
        assert_eq!(core_label(&info.cpus[0]), "AMD Ryzen 7");
    }

    #[test]
    fn groups_big_little() {
        let core = |cpu, name: &str, khz| CoreInfo { cpu, core: name.into(), max_khz: Some(khz) };
        let cs = clusters(&[
            core(0, "Cortex-A55", 1_800_000),
            core(1, "Cortex-A55", 1_800_000),
            core(2, "Cortex-A76", 2_200_000),
        ]);
        assert_eq!(cs.len(), 2);
        assert_eq!((cs[0].count, cs[1].cpus.clone()), (2, vec![2]));
    }

    #[test]
    fn meminfo_in_bytes() {
        let t = "MemTotal:        8245564 kB\nMemAvailable:    7000000 kB\n";
        assert_eq!(meminfo_bytes(t, "MemTotal"), Some(8_245_564 * 1024));
        assert_eq!(meminfo_bytes(t, "Nope"), None);
    }

    #[test]
    fn classifies_block_devices() {
        assert_eq!(block_kind("mmcblk0", Some("SD")), Some("sd"));
        assert_eq!(block_kind("mmcblk0", Some("MMC")), Some("emmc"));
        assert_eq!(block_kind("nvme0n1", None), Some("nvme"));
        assert_eq!(block_kind("zram0", None), None);
    }
}
