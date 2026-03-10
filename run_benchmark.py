import argparse
import sys
from rpx import RPXDataset
from rpx.tasks import (RelativePoseBenchmark, VisualGroundingBenchmark, 
                       DepthEstimationBenchmark, ObjectTrackingBenchmark,
                       NovelViewSynthesisBenchmark)

def main():
    parser = argparse.ArgumentParser(description="RPX Robot Perception Benchmark Suite")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of the RPX dataset")
    parser.add_argument("--task", type=str, required=True, 
                        choices=['relative_pose', 'grounding', 'depth', 'tracking', 'nvs'],
                        help="Task to benchmark")
    parser.add_argument("--scenes", type=str, nargs='*', help="Scenes to benchmark (default: all)")
    parser.add_argument("--phases", type=str, nargs='*', help="Phases to benchmark (default: all)")
    
    # In a real implementation, we would import the model here
    # or expose an interface for users to pass their models.
    
    args = parser.parse_args()
    
    print(f"Initializing RPX Dataset from: {args.root_dir}")
    dataset = RPXDataset(root_dir=args.root_dir, scenes=args.scenes, phases=args.phases)
    
    # Initialize appropriate benchmark
    if args.task == 'relative_pose':
        benchmark = RelativePoseBenchmark(dataset)
    elif args.task == 'grounding':
        benchmark = VisualGroundingBenchmark(dataset)
    elif args.task == 'depth':
        benchmark = DepthEstimationBenchmark(dataset)
    elif args.task == 'tracking':
        benchmark = ObjectTrackingBenchmark(dataset)
    elif args.task == 'nvs':
        benchmark = NovelViewSynthesisBenchmark(dataset)
        
    print(f"Running {args.task} benchmark...")
    # results = benchmark.evaluate(your_model)
    # print(results)
    print("Benchmark completed. Set a model to see results.")

if __name__ == "__main__":
    main()
