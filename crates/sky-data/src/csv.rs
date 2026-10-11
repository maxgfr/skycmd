//! Tiny RFC 4180 CSV parser (quoted fields, doubled quotes, embedded newlines).

/// Parses CSV text into rows of fields.
pub fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                '"' => in_quotes = false,
                _ => field.push(c),
            }
            continue;
        }
        match c {
            '"' => in_quotes = true,
            ',' => row.push(std::mem::take(&mut field)),
            '\r' => {}
            '\n' => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_quotes_and_newlines() {
        let rows = parse_csv("text,category\n\"a, \"\"b\"\"\",x\r\nplain,y\n\"multi\nline\",z");
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[1], vec!["a, \"b\"", "x"]);
        assert_eq!(rows[2], vec!["plain", "y"]);
        assert_eq!(rows[3], vec!["multi\nline", "z"]);
    }
}
