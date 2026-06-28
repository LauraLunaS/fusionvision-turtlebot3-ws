from roboflow import Roboflow

rf = Roboflow(api_key="UKdNPLg80oX345o0LtIo")  

project = rf.workspace("material-identification").project("garbage-classification-3")
dataset = project.version(2).download(
    model_format="yolov8",
    location="test_data/dataset"
)

print(f"\nClasses: {dataset.classes}")
print(f"Localização: {dataset.location}")