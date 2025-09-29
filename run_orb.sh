#!/bin/bash
truncate -s 0 /var/log/orb.log
truncate -s 0 cd /root/FyersORB/orb.log
cd /root/FyersORB
source venv/bin/activate
python -u main.py run 2>&1 | tee -a orb.log