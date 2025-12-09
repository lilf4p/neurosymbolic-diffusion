import os
import importlib


def get_all_models():
    models_dir = os.path.join(os.path.dirname(__file__))
    # Exclude diffusion models as they have their own training scripts
    excluded = ["indep_diffusion", "indep_diffusion_cfg"]
    return [
        model.split(".")[0]
        for model in os.listdir(models_dir)
        if not model.find("__") > -1 and "py" in model and os.path.isfile(os.path.join(models_dir, model))
        and model.split(".")[0] not in excluded
    ]


# Lazy loading of model classes to avoid circular imports
names = {}
_models_loaded = False

def _load_models():
    global names, _models_loaded
    if _models_loaded:
        return
    for model in get_all_models():
        mod = importlib.import_module("models." + model)
        class_name = {x.lower(): x for x in mod.__dir__()}[model.replace("_", "")]
        names[model] = getattr(mod, class_name)
    _models_loaded = True


def get_model(args, encoder, decoder, n_images, c_split):
    _load_models()
    if args.model == "cext":
        return names[args.model](encoder, n_images=n_images, c_split=c_split)
    elif args.model in [
        "mnistdpl",
        "mnistsl",
        "mnistindep",
        "mnistltn",
        "kanddpl",
        "kandltn",
        "kandpreprocess",
        "kandclip",
        "minikanddpl",
        "mnistpcbmdpl",
        "mnistpcbmsl",
        "mnistpcbmltn",
        "mnistclip",
        "sddoiadpl",
        "sddoiacbm",
        "sddoialtn",
        "presddoiadpl",
        "boiadpl",
        "mnistcbm",
        "boiacbm",
        "boialtn",
        "kandcbm",
        "mnistnn",
        "kandnn",
        "sddoiann",
        "sddoiaclip",
        "boiann",
        "xorcbm",
        "xornn",
        "xordpl",
        "mnmathnn",
        "mnmathcbm",
        "mnmathdpl",
        "mnistflow",
        "mnistmoe",
        "mnistsl_noisy",
        "mnistsl_indep",
        "mnistsl_flow",
        "mnistsl_adaptive",
        "mnistsl_proto",
    ]:
        return names[args.model](
            encoder, n_images=n_images, c_split=c_split, args=args
        )  # only discriminative
    else:
        return names[args.model](
            encoder, decoder, n_images=n_images, c_split=c_split, args=args
        )
