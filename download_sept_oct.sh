#!/bin/bash
# Download data from Sep 1, 2025 to Oct 31, 2025
for month in 09 10; do
    for day in $(seq -w 1 31); do
        date="2025-${month}-${day}"
        # Check if date is valid
        if date -j -f "%Y-%m-%d" "$date" "+%Y-%m-%d" >/dev/null 2>&1; then
            filename="subwaydatanyc_${date}_csv.tar.xz"
            url="https://subwaydata.nyc/data/${filename}"
            if [[ ! -f "data/${filename}" ]]; then
                echo "Downloading $filename..."
                curl -s -L -o "data/${filename}" "$url"
            fi
        fi
    done
done
echo "Download complete!"
