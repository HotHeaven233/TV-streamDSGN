#!/usr/bin/env python3

import sys

# Reuse exactly the same profiler implementation as Full.
# Only change the displayed variant name.
if "--variant_name" not in sys.argv:
    sys.argv.extend(
        [
            "--variant_name",
            "Light",
        ]
    )

from profile_full_components import main


if __name__ == "__main__":
    main()
