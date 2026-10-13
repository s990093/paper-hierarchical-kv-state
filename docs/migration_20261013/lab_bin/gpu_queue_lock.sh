#!/usr/bin/env bash
# GPU 佇列互斥鎖。
# 🔴 2026-09-18 踩到：先前用「等某個 PID 消失」來串接 GPU 佇列，抓到的卻是 runsh
#    包裝行程，它先結束 → 後面整批在 GPU 還被佔用時啟動，13 個步驟全部以
#    rc=5（GPU 沒空間）或 rc=2（guard 不乾淨）失敗。
# 改用 flock：同一時間只有一個 GPU 佇列拿得到鎖，其餘排隊等待。
exec 9>/var/tmp/tiara_gpu.lock
flock 9
exec "$@"
