//! Llama-3.1 chat prompt, as N-ATLaS's `tokenizer_config.json` template
//! renders a single system + user turn without tools.

pub const EOS: [u32; 3] = [128_001, 128_008, 128_009];

/// Rendered prompt text; `<|begin_of_text|>` is included, so encode it
/// without adding special tokens.
pub fn prompt(system: &str, user: &str) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
         Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{}<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n",
        system.trim(),
        user.trim()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_template_shape() {
        let p = prompt("", " Bawo ni? ");
        assert!(p.starts_with("<|begin_of_text|><|start_header_id|>system"));
        assert!(p.contains("Today Date: 26 Jul 2024\n\n<|eot_id|>"));
        assert!(p.contains("user<|end_header_id|>\n\nBawo ni?<|eot_id|>"));
        assert!(p.ends_with("assistant<|end_header_id|>\n\n"));
    }
}
