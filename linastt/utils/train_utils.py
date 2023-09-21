from linastt.utils.misc import remove_commonprefix
    
def args_to_str(args, ignore = [
        "gpus", "gpu"
        ]):
    if not isinstance(args, dict):
        args = args.__dict__

    s = "_".join(("{}-{}".format("".join([a[0] for a in k.replace("-","_").split("_")]),
            {True: 1, False: 0}.get(v, str(v).replace("/","_"))
        )) for k,v in sorted(args.items())
        if k not in ignore
    )
    while "__" in s:
        s = s.replace("__","_")
    return s
    
def dataset_pseudos(trainset, validset):
    train_folders = sorted(trainset.split(","))
    valid_folders = sorted(validset.split(","))
    all_folders = train_folders + valid_folders
    all_folders = remove_commonprefix(all_folders, "/")
    train_folders = all_folders[:len(train_folders)]
    valid_folders = all_folders[len(train_folders):]
    def base_folder(f):
        f = f.split("/")[0].split("\\")[0]
        if len(f.split("-")) > 1:
            f = "".join([s[0] for s in f.split("-")])
        return f
    train_base_folders = set(base_folder(f) for f in train_folders)
    valid_base_folders = set(base_folder(f) for f in valid_folders)
    train_folders = sorted(list(set([
        base_folder(f.replace("/","_")) if base_folder(f) in valid_base_folders else base_folder(f)
        for f in train_folders
    ])))
    valid_folders = sorted(list(set([
        base_folder(f.replace("/","_")) if base_folder(f) in train_base_folders else base_folder(f)
        for f in valid_folders
    ])))
    return "t-"+"-".join(train_folders), "v-"+"-".join(valid_folders)
