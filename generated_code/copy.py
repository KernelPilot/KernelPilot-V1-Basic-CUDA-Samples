import os
import shutil

# Define the number of tasks and the target folder
num_tasks = 20  # adjust this value based on your actual number of tasks
target_folder = "/home/wentaochen/cuda_gen/gen_20/gen"

# Create the target folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

for i in range(1, num_tasks + 1):
    task_folder = f"/home/wentaochen/cuda_gen/gen_20/task{i}"
    if not os.path.isdir(task_folder):
        continue

    # Find .cu file in task{i}
    cu_files = [f for f in os.listdir(task_folder) if f.endswith(".cu")]
    if not cu_files:
        continue

    # Use the first .cu file found
    source_path = os.path.join(task_folder, cu_files[0])
    target_path = os.path.join(target_folder, f"{i}.cu")

    shutil.copyfile(source_path, target_path)
