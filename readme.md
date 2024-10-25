you can pip install most packages needed in the importing part, 
however, you need to install pytorch with cuda enabled:
    a) first uninstall any pytorch version you already have ( pip uinstall torch)
    b) go here https://pytorch.org/get-started/locally 
    c) go to cmd, type: nvcc --version 
        - you can your cuda version, ex: 12.1
    d) in the website, choose the correct options and copy paste the run this command part, thats how you can install pytorch with cuda support
    e) you can check if you have correctly installed that by running check.py here

prepare the video, make sure that the name of the video is set correctly in the code also

IMPORTANT: 
    Just run the program, detected events are recorder in the folder detected_events with image and timestamp,
    frames without events are not saved.

    detector = GPUSuddenAppearanceDetector(
        threshold_pixel_diff=300,    # Adjust sensitivity
        min_area=100,              # Minimum size of changes to detect
        min_confidence=0.6,        # Minimum confidence threshold
        batch_size=1500,             # Reduced batch size for better stability
        use_gpu=True,             # Enable GPU acceleration
        output_dir="detected_events"  # Output directory for detections
    )

The threshold pixel diff. Lowering it makes it more sensitive to sudden changes, you can adjust to see 