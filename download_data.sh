#!/bin/bash
# Download data from Nov 1, 2025 to Jan 20, 2026
start_date="2025-11-01"
end_date="2026-01-20"
current="$start_date"

while [[ "$current" < "$end_date" ]] || [[ "$current" == "$end_date" ]]; do
    filename="subwaydatanyc_${current}_csv.tar.xz"
    url="https://subwaydata.nyc/data/${filename}"
    if [[ ! -f "data/${filename}" ]]; then
        echo "Downloading $filename..."
        curl -s -L -o "data/${filename}" "$url"
    else
        echo "Already have $filename"
    fi
    current=$(date -j -v+1d -f "%Y-%m-%d" "$current" "+%Y-%m-%d")
done
echo "Download complete!"
