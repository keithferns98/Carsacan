import pixellib
from pixellib.tune_bg import alter_bg
change_bg = alter_bg(model_type = "pb")
change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
change_bg.change_bg_img(f_image_path = "view1.jpeg",b_image_path = "back1.jpg", output_image_name="new_img1.jpg")
