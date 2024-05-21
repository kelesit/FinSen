from transformers import AutoModel

def print_trainable_parameters(model:"AutoModel"):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable parameters: {trainable_params} || all parameters: {all_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}"
    )