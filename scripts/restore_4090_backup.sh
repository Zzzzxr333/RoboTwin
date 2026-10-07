#!/usr/bin/env bash
# Run as root on Ubuntu 22.04 x86_64 after copying verified archives to this machine.
set -euo pipefail
BACKUP="${1:?Usage: bash restore_4090_backup.sh /path/to/4090_migration_20261007}"
cd "$BACKUP"
sha256sum -c SHA256SUMS
for path in /root/autodl-tmp/RoboTwin /root/autodl-tmp/conda_envs/RoboTwin /opt/pi05; do
  if [[ -e "$path" ]]; then echo "Target already exists: $path. Restore to a fresh instance to avoid overwriting work." >&2; exit 1; fi
done
apt-get update
apt-get install -y git ffmpeg tmux libvulkan1 vulkan-tools libegl1 libgl1 libglib2.0-0 libglfw3 libxrender1 libxext6 libgomp1 build-essential ninja-build
for archive in source_and_configs robotwin_environment pi05_environment cuda_toolkit assets datasets base_weights trained_checkpoints evaluation; do
  tar -xzf "$archive.tar.gz" -C /
done
git clone --no-checkout RoboTwin.bundle /root/autodl-tmp/migration_git_restore
git -C /root/autodl-tmp/migration_git_restore remote set-url origin https://github.com/Zzzzxr333/RoboTwin.git
mv /root/autodl-tmp/migration_git_restore/.git /root/autodl-tmp/RoboTwin/.git
rmdir /root/autodl-tmp/migration_git_restore
git -C /root/autodl-tmp/RoboTwin reset --mixed HEAD
# The source archive already contains the submodule files; initialize Git metadata separately.
git clone --no-checkout XPolicyLab.bundle /root/autodl-tmp/xpolicy_git_restore
git -C /root/autodl-tmp/xpolicy_git_restore remote set-url origin https://github.com/Zzzzxr333/XPolicyLab.git
mv /root/autodl-tmp/xpolicy_git_restore/.git /root/autodl-tmp/RoboTwin/XPolicyLab/.git
rmdir /root/autodl-tmp/xpolicy_git_restore
git -C /root/autodl-tmp/RoboTwin/XPolicyLab reset --mixed HEAD
source /root/autodl-tmp/RoboTwin/scripts/activate_autodl.sh
python scripts/verify_migration.py
echo 'Run the render and policy warmup commands in MIGRATION_README.md next.'
