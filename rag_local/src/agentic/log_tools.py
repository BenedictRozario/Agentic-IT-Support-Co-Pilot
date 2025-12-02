# log_tools.py
import re
from typing import Dict, Any

class LogTools:
    ERR_REGEX = re.compile(r"(ERR_[A-Z_]+)")
    PATH_REGEX = re.compile(r"(/[\\w\\/.\\-]+)")
    TIMESTAMP_REGEX = re.compile(r"\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}")

    def parse_log(self, log_text: str) -> Dict[str, Any]:
        codes = self.ERR_REGEX.findall(log_text)
        # find last error line
        last_error_line = None
        for line in reversed(log_text.splitlines()):
            if "ERROR" in line or "Exception" in line or self.ERR_REGEX.search(line):
                last_error_line = line
                break
        paths = self.PATH_REGEX.findall(log_text)[:3]
        timestamps = self.TIMESTAMP_REGEX.findall(log_text)
        return {
            "codes": list(dict.fromkeys(codes)),  # deduplicate preserving order
            "paths": paths,
            "last_error_line": last_error_line,
            "timestamps": timestamps
        }
