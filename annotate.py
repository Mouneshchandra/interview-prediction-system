def annotate_image(img_path,df):
    import cv2
    import pandas as pd

    # Load DataFrame
    df.columns = df.columns.str.strip()

    # Load an image
    image = cv2.imread(img_path)
    # Annotation settings
    font = cv2.FONT_HERSHEY_COMPLEX   # Formal font
    thickness = 1
    scale = 0.6
    x, y = 20, 40

    # ==================== 1. Likert Scale ====================
    # likert_score = float(df["hire_likert"][186])

    img_h, img_w = image.shape[:2]
    bar_x = img_w - 80
    bar_y = 50
    bar_height = 300
    bar_width = 20
    step = bar_height // 7

    # Likert scale colors
    scale_colors = [
        (0, 0, 150), (0, 90, 180), (0, 150, 150),
        (0, 150, 90), (0, 120, 0), (120, 60, 0), (80, 0, 80)
    ]

    for i in range(7):
        start_y = bar_y + (6 - i) * step
        end_y = start_y + step
        cv2.rectangle(image, (bar_x, start_y), (bar_x + bar_width, end_y), scale_colors[i], -1)
        cv2.putText(image, str(i+1), (bar_x + 30, start_y + step // 2),
                    font, 0.5, (255, 255, 255), 1)
    avg_likert = df["hire_likert"].mean()
    # Candidate Likert marker
    marker_y = int(bar_y + bar_height - (avg_likert - 1) * step - step / 2)
    cv2.line(image, (bar_x - 20, marker_y), (bar_x + bar_width + 20, marker_y), (255, 255, 255), 2)
    cv2.putText(image, f"Video Signal: {avg_likert:.2f}",
                (bar_x - 170, marker_y), font, 0.5, (10,10,10), 1)


    # ==================== 2. OCEAN Traits ====================
    traits = ["openness_level", "conscientiousness_level", "extraversion_level",
              "agreeableness_level", "neuroticism_level"]
   
    avg_traits = {trait:df[trait].mean() for trait in traits}
    trait_names = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


    # Trait label colors (BGR)
    trait_colors = [
        (115, 95, 0), (80, 127, 255), (220, 245, 245),
        (152, 251, 152), (96, 46, 4)
    ]

    y_offset = y
    gap = 20  # spacing between text and value

    for trait, name, t_color in zip(traits, trait_names, trait_colors):
        # score = int(df[trait][0])  # 1–3 levels
        avg_score = avg_traits[trait]
        avg_level = round(avg_score)

        # Color coding for levels
        if avg_level == 1:
            level_color = (0, 0, 255)   # Red
        elif avg_level == 2:
            level_color = (0, 165, 255) # Orange
        else:
            level_color = (0, 200, 0)   # Green

        # Draw trait name
        cv2.putText(image, f"{name}:", (x, y_offset), font, scale, t_color, thickness)

        # Measure text width
        (text_width, text_height), _ = cv2.getTextSize(f"{name}:", font, scale, thickness)

        # Place value right after text with a gap
        cv2.putText(image, f"{avg_score:.2f}", (x + text_width + gap, y_offset), font, scale, level_color, thickness)

        y_offset += 28

    return image