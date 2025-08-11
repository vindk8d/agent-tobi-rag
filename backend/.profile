#!/bin/bash
# Railway runtime profile for library paths

# Set library paths for WeasyPrint
export LD_LIBRARY_PATH="/nix/var/nix/profiles/default/lib:$LD_LIBRARY_PATH"
export GI_TYPELIB_PATH="/nix/var/nix/profiles/default/lib/girepository-1.0:$GI_TYPELIB_PATH"

# Additional paths that might be needed
for dir in /nix/store/*/lib; do
    if [ -d "$dir" ]; then
        export LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
    fi
done
