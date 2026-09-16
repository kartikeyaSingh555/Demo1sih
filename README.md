# IBVAP — Intelligent Border Video Analytics Platform

<p align="center">

  <img src="https://img.shields.io/badge/AI-Computer%20Vision-blue" />
  <img src="https://img.shields.io/badge/YOLO-Object%20Detection-green" />
  <img src="https://img.shields.io/badge/ArcFace-Face%20Recognition-purple" />
  <img src="https://img.shields.io/badge/ANPR-License%20Plate%20Recognition-orange" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688" />
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB" />
  <img src="https://img.shields.io/badge/MongoDB-Database-47A248" />
  <img src="https://img.shields.io/badge/Python-3.12.7-yellow" />
  <img src="https://img.shields.io/badge/Status-Prototype-success" />

</p>

<p align="center">
  <b>AI-Powered Border Intelligence, Threat Detection & Real-Time Surveillance</b>
</p>

---

## 📌 Overview

**IBVAP (Intelligent Border Video Analytics Platform)** is an AI-powered surveillance and border intelligence platform designed to continuously analyze CCTV, USB cameras, RTSP streams, IP cameras, and finite surveillance videos.

The platform combines:

- Computer Vision
- Object Detection
- Multi-Object Tracking
- Face Recognition
- Automatic Number Plate Recognition (ANPR)
- Weapon Detection
- UAV/Drone Detection
- Physical Fence Monitoring
- Camera Health Monitoring
- Threat Intelligence
- Event Correlation
- Real-Time Alerts
- Audio Siren
- Evidence Capture
- Incident Management
- Role-Based Access Control
- Cybersecurity
- Blockchain-based evidence integrity architecture

The system is designed around a **video-first architecture**, where smooth video playback/capture is separated from computationally expensive AI inference.

---

# 🎯 Problem Statement

Traditional CCTV surveillance systems primarily depend on human operators continuously watching multiple camera feeds.

This creates several challenges:

- Large numbers of cameras are difficult to monitor simultaneously.
- Important events can be missed.
- Repetitive monitoring causes operator fatigue.
- Suspicious objects may not be detected quickly.
- Unauthorized personnel may go unnoticed.
- Unknown vehicles may enter restricted areas.
- Drones/UAVs may approach sensitive zones.
- Physical fence breaches may not be immediately identified.
- Camera failures may remain unnoticed.
- Large volumes of video make post-incident investigation difficult.
- Evidence integrity must be maintained during investigations.

IBVAP addresses these challenges by adding an intelligent AI perception and threat-analysis layer on top of surveillance infrastructure.

---

# 🚀 Key Objectives

The major objectives of IBVAP are:

1. Continuously monitor border surveillance cameras.
2. Detect people and vehicles automatically.
3. Track objects across video frames.
4. Identify authorized personnel using face recognition.
5. Detect unknown personnel.
6. Recognize vehicle license plates using ANPR.
7. Identify authorized and unknown vehicles.
8. Detect weapon-like objects.
9. Detect UAVs/drones.
10. Monitor physical border fences.
11. Detect fence damage and potential breaches.
12. Monitor CCTV camera health.
13. Correlate multiple security signals.
14. Generate threat scores.
15. Generate real-time alerts.
16. Activate an audio siren for qualifying high-severity events.
17. Capture evidence for important incidents.
18. Maintain incident timelines.
19. Provide a centralized command center.
20. Protect surveillance information using cybersecurity controls.
21. Provide tamper-evident evidence verification using cryptographic hashing/blockchain architecture.

---

# 🧠 System Architecture

```text
                         BORDER CCTV / VIDEO SOURCES
                                   │
                  ┌────────────────┼────────────────┐
                  │                │                │
                 USB              RTSP          IP CAMERA
                  │                │                │
                  └────────────────┼────────────────┘
                                   │
                                   ↓
                         VIDEO CAPTURE LAYER
                                   │
                     Latest Frame / Buffer
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ↓                                   ↓
        SMOOTH VIDEO STREAM                    AI ML WORKER
        25–30 FPS target                         3–5 FPS
                 │                                   │
                 │                    ┌──────────────┼──────────────┐
                 │                    │              │              │
                 │                    ↓              ↓              ↓
                 │              Object Detection  Tracking      Face
                 │                    │              │           Recognition
                 │                    │              │              │
                 │                    ├──────────────┼──────────────┤
                 │                    │              │              │
                 │                    ↓              ↓              ↓
                 │                  ANPR          Weapons        UAV/Drone
                 │                    │              │              │
                 │                    ├──────────────┼──────────────┤
                 │                    │              │
                 │                    ↓              ↓
                 │                 Fence       Camera Health
                 │                    │              │
                 └────────────────────┴──────────────┘
                                      │
                                      ↓
                              UNIFIED ML EVENTS
                                      │
                                      ↓
                              EVENT CORRELATION
                                      │
                                      ↓
                              THREAT ENGINE
                                      │
                     ┌────────────────┼────────────────┐
                     │                │                │
                     ↓                ↓                ↓
                  ALERTS          INCIDENTS         EVIDENCE
                     │                │                │
                     ↓                ↓                ↓
                  SIREN          TIMELINE         SNAPSHOTS
                     │                │                │
                     └────────────────┼────────────────┘
                                      ↓
                              COMMAND CENTER
                                      │
                     ┌────────────────┼────────────────┐
                     ↓                ↓                ↓
                 Dashboard       Live Monitoring     Analytics
                     │                │                │
                     ↓                ↓                ↓
                Watchlist          Alerts          AI Assistant
