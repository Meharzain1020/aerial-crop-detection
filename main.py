import torch
from torch.utils.data import DataLoader
from src.data_utils import generate_synthetic_crop_data, AerialCropDataset
from src.model import build_crop_detector
from src.train import train_model
from src.evaluate import run_evaluation

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    print("[1/5] Setting up data paths and generating synthetic data...")
    raw_data_dir = "data/raw"
    generate_synthetic_crop_data(raw_data_dir, num_samples=10)

    print("[2/5] Preparing DataLoaders...")
    dataset = AerialCropDataset(data_dir=raw_data_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    print("[3/5] Instantiating Faster R-CNN with FPN Backbone...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_crop_detector(num_classes=3)

    print("[4/5] Running Training Pipeline...")
    trained_model = train_model(
        model=model, 
        dataloader=dataloader, 
        epochs=7, 
        device=device,
        output_weights="outputs/weights/fine_tuned_crop_detector.pth"
    )

    print("[5/5] Running Evaluation & Visualizations...")
    run_evaluation(
        model=trained_model, 
        dataloader=dataloader, 
        device=device, 
        output_csv="outputs/results_summary.csv"
    )

    print("\nExecution complete. All outputs saved to 'outputs/'.")

if __name__ == "__main__":
    main()