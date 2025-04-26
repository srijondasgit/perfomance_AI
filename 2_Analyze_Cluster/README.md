# Cluster Performance Prediction Demo

This project simulates and analyzes system performance data.

- We generate synthetic data for **5 clusters**, each with **25 computers**.
- Each entry includes a **timestamp**, **CPU**, **Memory**, and **Disk** usage, along with a **performance label**.

## What the Training Script Does

- **Checks** if any **cluster** or **computer** showed **HighLoad** in the **last 10 minutes**.
- **Trains** a lightweight GPT-2 model on the performance data.
- **Predicts** whether a computer's current performance is likely to be poor based on system metrics.

This is a basic prototype for real-time performance monitoring and prediction using language models.
