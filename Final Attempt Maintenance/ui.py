<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aircraft Maintenance Prediction</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        .slider-container {
            margin: 10px 0;
        }
        .slider {
            width: 100%;
        }
        .slider-label {
            display: inline-block;
            width: 200px;
        }
    </style>
</head>
<body>
    <div style="text-align:center;">
        <h2>Aircraft Maintenance Rating Prediction</h2>

        <form method="POST">
            {% for feature, range in feature_info.items() %}
            <div class="slider-container">
                <label for="{{ feature }}" class="slider-label">{{ feature.replace('_', ' ').title() }}:</label>
                <input type="range" name="{{ feature }}" min="{{ range[0] }}" max="{{ range[1] }}" class="slider" value="{{ range[0] }}" step="any">
                <span id="{{ feature }}-value"> {{ range[0] }} </span>
            </div>
            {% endfor %}
            <button type="submit">Predict Rating</button>
        </form>

        {% if rating is not none %}
        <h3>Predicted Rating: {{ rating }}</h3>
        {% endif %}
    </div>

    <script>
        // Update value display for sliders
        const sliders = document.querySelectorAll('.slider');
        sliders.forEach(slider => {
            slider.oninput = function() {
                document.getElementById(slider.name + '-value').textContent = slider.value;
            }
        });
    </script>
</body>
</html>
