from depth_benchmark.manager import BenchmarkManager
from depth_benchmark.models.zoe_depth import ZoeDepth
from depth_benchmark.models.depth_anything import DepthAnything
from depth_benchmark.utils.dataset_loader import DatasetLoader
from depth_benchmark.utils.evaluator import Evaluator


def main():
    """Main function to set up and run the benchmark."""
    print("Initializing benchmark framework...")

    # 1. Instantiate all models to be tested
    # In a real scenario, you might pass 'cuda' if a GPU is available
    models_to_test = [
        ZoeDepth(device='cuda'),
        DepthAnything(device='cuda'),
    ]

    # 2. Instantiate the utility classes
    dataset = DatasetLoader(dataset_path='./data')
    evaluator = Evaluator()

    # 3. Instantiate and run the Benchmark Manager
    manager = BenchmarkManager(
        models=models_to_test,
        dataset_loader=dataset,
        evaluator=evaluator
    )

    # 4. Execute the benchmark and print the report
    manager.run_all()
    manager.report_results()


if __name__ == "__main__":
    main()