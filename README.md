This is an example on how to create a RAG chatbot using Llama-Index, Docker for containerization, Mlflow, Prometheus and Grafana, all running in Docker.
For running this example in docker using local sources i used a small LLM from Hugging Face: meta-llama/Llama-3.2-1B-Instruct

Simple HTML for our testing:

![Screenshot 2025-06-24 at 1 44 40 PM](https://github.com/user-attachments/assets/ab593744-6d52-4ed7-93fb-d783536f8667)

Docker:

![Screenshot 2025-06-24 at 1 45 03 PM](https://github.com/user-attachments/assets/0508a8ee-36be-46c0-92af-473e399c07cc)

Mlflow:

![Screenshot 2025-06-24 at 1 45 15 PM](https://github.com/user-attachments/assets/2f7d3630-1c93-4270-910d-35d3085a6c98)

Use of Promitheus for extracting the metrics needed to feed Grafana:

![Screenshot 2025-06-24 at 1 07 54 PM](https://github.com/user-attachments/assets/18305d98-a8ac-4d2b-9884-b0dc1f0d7a42)


Grafana dashboard for monitoring of our model:

![Screenshot 2025-06-24 at 1 45 46 PM](https://github.com/user-attachments/assets/99f8cef3-2d8e-4464-b0a6-dd16039d80f6)


This is a proof of concept scenario application and it does not represent a total solution.
