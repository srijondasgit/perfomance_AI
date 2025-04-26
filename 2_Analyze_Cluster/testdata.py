import random
from datetime import datetime, timedelta

# Configurable parameters
NUM_CLUSTERS = 5
COMPUTERS_PER_CLUSTER = 25
TOTAL_ENTRIES = 5000

def generate_entry():
    # Generate timestamp within the past 24 hours
    now = datetime.now()
    minutes_ago = random.randint(0, 1440)
    timestamp = (now - timedelta(minutes=minutes_ago)).isoformat()

    # Select cluster and computer
    cluster_id = f"cluster-{random.randint(1, NUM_CLUSTERS)}"
    computer_id = f"comp-{random.randint(1, COMPUTERS_PER_CLUSTER):02}"

    # Generate performance metrics
    cpu = random.randint(5, 100)
    memory = random.randint(1, 16)
    disk = random.randint(5, 100)

    # Assign label based on CPU and Disk usage
    if cpu > 85 or disk > 90:
        label = "HighLoad"
    elif cpu > 50 or disk > 70:
        label = "MediumLoad"
    else:
        label = "LowLoad"

    return (f"[{timestamp}] Cluster: {cluster_id}, Computer: {computer_id}, "
            f"CPU Load: {cpu}%, Memory: {memory}GB, Disk: {disk}% -> Label: {label}")

# Write synthetic data to train.txt
with open("train.txt", "w") as f:
    for _ in range(TOTAL_ENTRIES):
        f.write(generate_entry() + "\n")
