"""Seed the database with realistic demo jobs so the app works before scraping."""

from src.db import init_db, insert_job
from src.embeddings import embed_texts

DEMO_JOBS = [
    {
        "title": "Computer Vision Engineer",
        "company": "TRACKAI",
        "location": "Copenhagen, Denmark",
        "description": "Develop player-tracking algorithms for professional football. Experience with PyTorch, OpenCV, and multi-object tracking required.",
        "url": "https://example.com/cv-engineer-trackai",
        "source": "demo",
        "salary": "€60k - €75k",
        "job_type": "Full-time",
        "skills": "PyTorch, OpenCV, MOT, Python, Football",
        "posted_date": "2026-07-20",
    },
    {
        "title": "Senior Machine Learning Engineer - Sports Analytics",
        "company": "StatsBomb",
        "location": "London, UK",
        "description": "Build event data models for football analytics. Strong Python, xG models, and data engineering skills.",
        "url": "https://example.com/ml-statsbomb",
        "source": "demo",
        "salary": "£85k - £110k",
        "job_type": "Full-time",
        "skills": "Python, Football Analytics, xG, ML",
        "posted_date": "2026-07-18",
    },
    {
        "title": "Research Scientist - Computer Vision",
        "company": "SoccerVision Lab",
        "location": "Milan, Italy",
        "description": "Research in pose estimation and camera calibration for broadcast soccer video. PhD preferred.",
        "url": "https://example.com/research-soccervision",
        "source": "demo",
        "salary": "",
        "job_type": "Full-time",
        "skills": "Computer Vision, Pose Estimation, Camera Calibration",
        "posted_date": "2026-07-15",
    },
    {
        "title": "Software Engineer - Vision Systems",
        "company": "Veo Technologies",
        "location": "Copenhagen, Denmark",
        "description": "Automatic camera systems for amateur football. Work on embedded vision, streaming, and edge AI.",
        "url": "https://example.com/software-veo",
        "source": "demo",
        "salary": "DKK 45,000/mo",
        "job_type": "Full-time",
        "skills": "C++, Python, Embedded Vision, Streaming",
        "posted_date": "2026-07-12",
    },
    {
        "title": "AI Engineer - Autonomous Robots",
        "company": "Roco Robotics",
        "location": "Berlin, Germany",
        "description": "Perception stack for autonomous soccer robots. SLAM, sensor fusion, ROS2.",
        "url": "https://example.com/ai-roco",
        "source": "demo",
        "salary": "€70k - €90k",
        "job_type": "Full-time",
        "skills": "ROS2, SLAM, Sensor Fusion, Robotics",
        "posted_date": "2026-07-10",
    },
    {
        "title": "Data Scientist - Player Performance",
        "company": "Nordic Football Analytics",
        "location": "Stockholm, Sweden",
        "description": "Analyze tracking data to build performance metrics for clubs. Strong statistics and Python.",
        "url": "https://example.com/ds-nordic-football",
        "source": "demo",
        "salary": "SEK 55,000/mo",
        "job_type": "Full-time",
        "skills": "Statistics, Python, Sports Data, Tracking",
        "posted_date": "2026-07-08",
    },
    {
        "title": "MLOps Engineer",
        "company": "PlayrAI",
        "location": "Remote (Europe)",
        "description": "Deploy and monitor ML pipelines for youth football analysis. Docker, Kubernetes, AWS.",
        "url": "https://example.com/mlops-playrai",
        "source": "demo",
        "salary": "€65k - €80k",
        "job_type": "Remote",
        "skills": "Docker, Kubernetes, AWS, ML Pipelines",
        "posted_date": "2026-07-05",
    },
    {
        "title": "Camera Calibration Engineer",
        "company": "VizTrack",
        "location": "Amsterdam, Netherlands",
        "description": "Calibrate multi-camera setups for stadiums. Computer vision, MATLAB/C++, photogrammetry.",
        "url": "https://example.com/camera-viztrack",
        "source": "demo",
        "salary": "€58k - €72k",
        "job_type": "Full-time",
        "skills": "Camera Calibration, Computer Vision, C++",
        "posted_date": "2026-07-02",
    },
]


def seed_demo() -> dict:
    init_db()
    texts = [
        f"{j['title']}. {j['company']}. {j['location']}. {j['description']} {j['skills']}"
        for j in DEMO_JOBS
    ]
    embeddings = embed_texts(texts)
    new = 0
    for job, emb in zip(DEMO_JOBS, embeddings):
        if insert_job(job, embedding=emb):
            new += 1
    return {"new": new, "total_in_demo": len(DEMO_JOBS)}


if __name__ == "__main__":
    result = seed_demo()
    print(f"Seeded {result['new']} new demo jobs.")
