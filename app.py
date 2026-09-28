from flask import Flask, render_template, request, url_for,Response,jsonify,redirect,current_app
from werkzeug.utils import secure_filename
import new_ocean as no
import os
import pandas as pd
import shutil
import cv2
import numpy as np
import time
import threading
from llm_feedback import generate_feedback

app = Flask(__name__, template_folder=".")
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

#Global variables
streaming = True
latest_result = {"segment": 0, "score": None, "label": None, "annotated_path": None}
all_segment_results = []  # <-- new list to store all segment results
frame_buffer = []
video_frame_counter = 0
ANALYSIS_FRAME_INTERVAL = 10
segment_count = 1

# Global event to signal when a segment has finished processing
segment_ready = threading.Event()  # initially not set (cleared)
processing_lock = threading.Lock() 
frame_buffer_lock = threading.Lock()
final_annotated_path = None
video_source = 0  # default webcam
TRAIT_LABELS = (
    ("openness_signal", "Openness cue"),
    ("conscientiousness_signal", "Conscientiousness cue"),
    ("extraversion_signal", "Extraversion cue"),
    ("agreeableness_signal", "Agreeableness cue"),
    ("neuroticism_signal", "Neuroticism cue"),
)


def build_summary_payload(averages, final_avg_score, feedback, annotated_filename):
    trait_rows = []
    for key, label in TRAIT_LABELS:
        value = averages.get(key)
        if value is None or not pd.notna(value):
            continue
        numeric_value = float(value)
        trait_rows.append({
            "name": label,
            "score": f"{numeric_value:.2f}",
            "width": max(0, min(100, (numeric_value - 1) * 50)),
        })

    score_percent = None
    if final_avg_score is not None and pd.notna(final_avg_score):
        score_percent = max(0, min(100, (float(final_avg_score) - 1) / 6 * 100))

    return {
        "candidate": app.config.get("CANDIDATE_DATA", {}),
        "averages": averages,
        "trait_rows": trait_rows,
        "segment_rows": list(all_segment_results),
        "final_avg_score": final_avg_score,
        "score_percent": score_percent,
        "final_label": "Supplementary signal only",
        "annotated_image": annotated_filename,
        "feedback": feedback,
        "candidate_takeaways": feedback["candidate_takeaways"],
        "practice_steps": feedback["practice_steps"],
        "signal_note": feedback["signal_note"],
        "followup_note": feedback["followup_note"],
        "limits_note": feedback["limits_note"],
    }

@app.route('/')
def form():
    return render_template("candidate.html", values={}, error=None)


@app.route("/begin_interview", methods=["POST"])
def begin_interview():
    candidate_name = request.form.get("candidate_name", "").strip()
    role_title = request.form.get("role_title", "").strip()
    email = request.form.get("email", "").strip()
    experience = request.form.get("experience", "").strip()

    if not candidate_name or not role_title or request.form.get("video_consent") != "on":
        return render_template(
            "candidate.html",
            values=request.form,
            error="Enter the candidate name and role, and confirm consent before continuing.",
        ), 400

    if experience:
        try:
            experience_years = int(experience)
            if not 0 <= experience_years <= 60:
                raise ValueError
        except ValueError:
            return render_template(
                "candidate.html",
                values=request.form,
                error="Experience must be a whole number from 0 to 60.",
            ), 400
    else:
        experience_years = None

    app.config["CANDIDATE_DATA"] = {
        "name": candidate_name,
        "role": role_title,
        "email": email,
        "experience": experience_years,
    }
    app.config.pop("VIDEO_SOURCE_PATH", None)
    return redirect(url_for("interview"))


@app.route("/interview")
def interview():
    candidate = app.config.get("CANDIDATE_DATA")
    if not candidate:
        return redirect(url_for("form"))
    return render_template("interview.html", candidate=candidate)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_source
    if not app.config.get("CANDIDATE_DATA"):
        return jsonify({"error": "Enter candidate details before uploading a video."}), 403

    file = request.files.get('video_file')

    if not file:
        return jsonify({"error": "No video file uploaded"}), 400

    filename = secure_filename(file.filename or "")
    if not filename:
        return jsonify({"error": "The uploaded file must have a valid filename."}), 400
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    video_source = filepath  # use uploaded video instead of webcam
    print(f"📁 Uploaded video selected: {filepath}")

    # ✅ Save it in Flask config so start_processing can find it
    app.config["VIDEO_SOURCE_PATH"] = filepath
    print(f"📁 Uploaded video set as source: {filepath}")
    return jsonify({"status": "uploaded", "path": filepath}), 200


def process_segments(segment_duration=15):
    global streaming

    while streaming:
        time.sleep(segment_duration)  # wait for segment_duration seconds
        if not streaming:
            print("🛑 Processing stopped (thread exiting)")
            break  # ✅ exit loop immediately

        process_pending_segment()


def process_pending_segment():
    global frame_buffer, segment_count, latest_result, all_segment_results

    with processing_lock:
        with frame_buffer_lock:
            if not frame_buffer:
                return False
            frames = list(frame_buffer)
            frame_buffer.clear()

        current_segment = segment_count
        print(f"🟢 Processing Segment {current_segment} ({len(frames)} frames)")
        try:
            score, label, annotated_path = no.ocean_average(frames, current_segment)
            latest_result = {
                "segment": current_segment,
                "score": score,
                "label": label,
                "annotated_path": annotated_path,
                "error": None,
            }
            if score is not None:
                all_segment_results.append({
                    "segment": current_segment,
                    "score": score,
                    "label": label,
                })
        except Exception as error:
            latest_result = {
                "segment": current_segment,
                "score": None,
                "label": "Processing error",
                "annotated_path": None,
                "error": str(error),
            }
            print(f"❌ Segment {current_segment} failed: {error}")
        finally:
            segment_count += 1
            open("of_au.csv", "w").close()
            segment_ready.set()
        return True


def generate_frames():
    global streaming, video_source, frame_buffer, video_frame_counter

    print(f"🎥 Starting video stream from: {video_source}")

    camera = cv2.VideoCapture(video_source)
    print("Video runnning", video_source)
    if not camera.isOpened():
        print("❌ Error: Unable to access the camera or video source.")
        blank = cv2.imencode('.jpg', 255 * np.ones((480, 640, 3), np.uint8))[1].tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + blank + b'\r\n')
        streaming = False
        auto_stop_and_redirect()
        return

    while streaming:
        success, frame = camera.read()
        if not success:
            print("⚠️ Frame read failed — maybe video ended or camera disconnected.")

            # If it's a video file → stop and redirect automatically
            if not str(video_source).isdigit():
                print("✅ Uploaded video ended — auto-stop.")
                streaming = False
                camera.release()
                cv2.destroyAllWindows()
                process_pending_segment()
                auto_stop_and_redirect()
                return

            time.sleep(0.2)
            continue

        if not streaming:
            break

        if video_frame_counter % ANALYSIS_FRAME_INTERVAL == 0:
            with frame_buffer_lock:
                frame_buffer.append(frame)
        video_frame_counter += 1
        _, buffer = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    camera.release()
    cv2.destroyAllWindows()
    print("🛑 Stream closed.")


def auto_stop_and_redirect():
    """Auto stop when video file ends and compute summary."""
    global streaming, all_segment_results, final_annotated_path

    streaming = False
    print("⏹ Auto stop triggered (video ended)")
    open(r"of_au.csv", "w").close()

    # Capture final annotated image path
    if latest_result and latest_result.get("annotated_path"):
        final_annotated_path = latest_result["annotated_path"]
        print(f"🖼️ Final annotated image path: {final_annotated_path}")
    else:
        final_annotated_path = None
        print("⚠️ No annotated image found on auto-stop.")

    app.config["FINAL_ANNOTATED_PATH"] = final_annotated_path

    # Directly compute summary once the video finishes
    try:
        results_path = os.path.join("segment_results", "segment_ocean_hire.csv")
        if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
            error = (latest_result or {}).get("error")
            raise ValueError(error or "No valid face features were produced. Check the video and OpenFace setup.")
        df = pd.read_csv(results_path, skip_blank_lines=True)
        numeric_cols = [
            'openness_signal',
            'conscientiousness_signal',
            'extraversion_signal',
            'agreeableness_signal',
            'neuroticism_signal',
            'hire_likert'
        ]
        if df.empty:
            raise ValueError("No usable face-feature rows were produced from the video.")
        averages = df[numeric_cols].mean().round(2).to_dict()
        final_avg_score = round(df["hire_likert"].mean(), 2)
        final_label = "Supplementary signal only"
        feedback = generate_feedback(
            final_avg_score, final_label, averages,
            [result["score"] for result in all_segment_results],
        )

        annotated_filename = None
        if final_annotated_path and os.path.exists(final_annotated_path):
            annotated_filename = os.path.basename(final_annotated_path)

        # Save summary data in config
        app.config["SUMMARY_DATA"] = build_summary_payload(
            averages, final_avg_score, feedback, annotated_filename
        )
        app.config["AUTO_SUMMARY_READY"] = True  # ✅ Add this flag
        app.config["PROCESSING_FINISHED"] = True
        print("✅ Auto summary computed successfully")

    except Exception as e:
        print(f"⚠️ Auto summary error: {e}")
        app.config["SUMMARY_DATA"] = build_summary_payload(
            {}, None,
            generate_feedback(None, "No usable video signal", {}),
            None,
        )
        app.config["AUTO_SUMMARY_READY"] = True
        app.config["PROCESSING_FINISHED"] = True

@app.route('/check_auto_summary')
def check_auto_summary():
    if app.config.get("AUTO_SUMMARY_READY"):
        app.config["AUTO_SUMMARY_READY"] = False  # reset flag
        print("🔍 check_auto_summary called — ready =", True)
        return jsonify({"ready": True})
    print("🔍 check_auto_summary called — ready =", False)
    return jsonify({"ready": False})


@app.route('/get_results')
def get_results():
    global segment_ready, latest_result

    if app.config.get("PROCESSING_FINISHED"):
        return jsonify({"status": "finished"})
    if segment_ready.is_set():
        # Segment is ready → send it and reset event
        segment_ready.clear()
        return jsonify({**latest_result,"status":"ready"})
    else:
        # Segment not ready → return previous segment or "waiting"
        return jsonify({
            "segment": latest_result["segment"],
            "score": "-",
            "label": "-",
            "status": "waiting"
        })

@app.route('/start_processing', methods=["POST"])
def start_processing():
    global streaming, segment_count, all_segment_results, frame_buffer, video_source, video_frame_counter

    if not app.config.get("CANDIDATE_DATA"):
        return jsonify({"error": "Enter candidate details before starting an interview."}), 403

    source_mode = (request.get_json(silent=True) or {}).get("source", "")
    if source_mode == "upload":
        uploaded_video = app.config.get("VIDEO_SOURCE_PATH")
        if not uploaded_video or not os.path.exists(uploaded_video):
            return jsonify({"error": "Upload a video recording before starting analysis."}), 400
        video_source = uploaded_video
        print(f"🎞️ Using uploaded video: {video_source}")
    elif source_mode == "webcam":
        video_source = 0
        print("🎥 Using webcam as video source")
    else:
        return jsonify({"error": "Choose an uploaded recording or webcam as the source."}), 400

    streaming = True
    segment_count = 1
    video_frame_counter = 0
    with frame_buffer_lock:
        frame_buffer.clear()
    all_segment_results.clear()
    latest_result.update({"segment": 0, "score": None, "label": None, "annotated_path": None, "error": None})
    app.config["SUMMARY_DATA"] = None
    app.config["AUTO_SUMMARY_READY"] = False
    app.config["PROCESSING_FINISHED"] = False
    segment_ready.clear()
    results_path = os.path.join("segment_results", "segment_ocean_hire.csv")
    if os.path.exists(results_path):
        os.remove(results_path)
    open(r"of_au.csv", "w").close()

    # Start background thread if not already running
    if not hasattr(app, "segment_thread") or not app.segment_thread.is_alive():
        app.segment_thread = threading.Thread(target=process_segments, daemon=True)
        app.segment_thread.start()

    return jsonify({"status": "started"}), 200

@app.route('/stop_processing')
def stop_processing():
    global streaming
    streaming = False
    print("⏹ Streaming stopped")
    process_pending_segment()
    auto_stop_and_redirect()
    return redirect(url_for('summary'))

@app.route('/summary')
def summary():
    if not app.config.get("CANDIDATE_DATA"):
        return redirect(url_for("form"))

    cached_summary = app.config.get("SUMMARY_DATA")
    annotated_path = app.config.get("FINAL_ANNOTATED_PATH")

    if cached_summary:
        return render_template("review.html", **cached_summary)

    try:
        df = pd.read_csv(r"segment_results\segment_ocean_hire.csv", skip_blank_lines=True)

        numeric_cols = [
            'openness_signal',
            'conscientiousness_signal',
            'extraversion_signal',
            'agreeableness_signal',
            'neuroticism_signal',
            'hire_likert'
        ]

        if df.empty:
            raise ValueError("No usable face-feature rows were produced from the video.")
        averages = df[numeric_cols].mean().round(2).to_dict()
        final_avg_score = round(df["hire_likert"].mean(), 2)
        final_label = "Supplementary signal only"

        # 🧠 Generate feedback
        feedback = generate_feedback(
            final_avg_score, final_label, averages,
            [result["score"] for result in all_segment_results],
        )

        # Extract image filename if exists
        annotated_filename = None
        if annotated_path and os.path.exists(annotated_path):
            annotated_filename = os.path.basename(annotated_path)

        return render_template(
            "review.html",
            **build_summary_payload(averages, final_avg_score, feedback, annotated_filename),
        )

    except Exception as e:
        print("⚠️ Error computing summary:", e)
        return render_template(
            "review.html",
            **build_summary_payload(
                {}, None,
                generate_feedback(None, "No usable video signal", {}),
                None,
            ),
        )


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/reset_session')
def reset_session():
    global streaming, segment_count, frame_buffer, all_segment_results, latest_result, video_source

    print("🔄 Resetting interview session...")

    # Stop any ongoing streaming or processing
    streaming = False
    time.sleep(0.5)

    # Clear all global data
    segment_count = 1
    with frame_buffer_lock:
        frame_buffer.clear()
    all_segment_results.clear()
    latest_result = {"segment": 0, "score": None, "label": None, "annotated_path": None}
    video_source = 0
    app.config.pop("VIDEO_SOURCE_PATH", None)
    app.config.pop("FINAL_ANNOTATED_PATH", None)
    app.config.pop("SUMMARY_DATA", None)
    app.config.pop("CANDIDATE_DATA", None)
    app.config.pop("AUTO_SUMMARY_READY", None)
    app.config.pop("PROCESSING_FINISHED", None)

    # Remove temporary files if needed
    try:
        if os.path.exists("of_au.csv"):
            open("of_au.csv", "w").close()
        print("🧹 Temporary files cleared.")
    except Exception as e:
        print("⚠️ Cleanup error:", e)

    return redirect(url_for('form'))

if __name__ == '__main__':
    app.run(debug=False, port=int(os.environ.get("PORT", "5001")))