import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 Training Script')
    parser.add_argument('--model', type=str, default=None, help='Path to model file or pretrained weights')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to dataset YAML file')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer type (SGD, Adam, AdamW, etc.)')
    parser.add_argument('--lr0', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.93, help='SGD momentum/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Optimizer weight decay')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--patience', type=int, default=10, help='Epochs to wait for no improvement before early stopping')
    parser.add_argument('--device', type=str, default='0', help='Device to run on (e.g., cuda:0, cpu)')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loader workers')
    parser.add_argument('--save_period', type=int, default=50, help='Save checkpoint every X epochs')
    parser.add_argument('--cache', action='store_true', help='Cache dataset in RAM for faster training')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision training')
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data_dir,
        resume=args.resume,
        epochs=args.epochs,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        save_period=args.save_period,
        cache=args.cache,
        amp=args.amp
    )

if __name__ == '__main__':
    main()