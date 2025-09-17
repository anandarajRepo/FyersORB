#!/bin/bash
truncate -s 0 /var/log/orb.log
cd /root/FyersORB
source venv/bin/activate
python main.py run >> /var/log/orb.log 2>&1