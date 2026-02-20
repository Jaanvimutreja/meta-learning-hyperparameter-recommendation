import subprocess

print("🚀 Starting Full Meta-Learning Pipeline...\n")

print("1️⃣ Training base models...")
subprocess.run(["python", "main.py"])

print("\n2️⃣ Building meta dataset...")
subprocess.run(["python", "build_meta_dataset.py"])

print("\n3️⃣ Training meta model...")
subprocess.run(["python", "train_meta_model.py"])

print("\n🔥 Pipeline completed successfully!")
