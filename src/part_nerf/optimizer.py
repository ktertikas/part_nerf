from torch.optim import SGD, Adam, AdamW, Optimizer


def build_optimizer(opt_config, model_params) -> Optimizer:
    lr = opt_config.get("lr")
    opt_type = opt_config.get("type")
    momentum = opt_config.get("momentum", 0.0)
    weight_decay = opt_config.get("weight_decay", 0.0)
    if opt_type == "Adam":
        return Adam(model_params, lr=lr, weight_decay=weight_decay)
    if opt_type == "AdamW":
        return AdamW(model_params, lr=lr, weight_decay=weight_decay)
    if opt_type == "SGD":
        return SGD(model_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise NotImplementedError(
        f"Optimizer of type {opt_config['type']} is not implemented"
    )
