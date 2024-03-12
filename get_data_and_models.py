
def get_cifar10(datadir):

   from torchvision.datasets import CIFAR10

   dataset = CIFAR10(root=datadir, train=True, download=True)



def get_hf_dataset(dataset_name_and_config, **kwargs):

   from datasets import load_dataset

   dataset = load_dataset(*dataset_name_and_config, **kwargs)


def get_hf_model(model_name):

   from transformers import AutoTokenizer, AutoModelForSequenceClassification

   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForSequenceClassification.from_pretrained(model_name)


def get_llama_tokenizer():

   from transformers import LlamaTokenizerFast

   tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")


def main():

   print("Downloading data for image classification benchmark...")
   get_cifar10("./benchmarks/image_classification/data")

   print("Downloading data for language model benchmark...")
   get_hf_dataset(("glue","cola"))

   print("Downloading data for large language model benchmark...")
   get_hf_dataset(("HuggingFaceH4/ultrachat_200k",), split="train_gen")

   print("Downloading model and tokenizer for language model benchmark...")
   get_hf_model("google-bert/bert-base-cased")

   print("Downloading model and tokenizer for large language model benchmark...")
   get_llama_tokenizer()


if __name__=='__main__':
   main()
