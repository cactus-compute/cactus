def run(args):
    import json
    from .. import Mirage
    options = json.loads(args.options)
    if not isinstance(options, dict):
        raise SystemExit("--options must be a JSON object")
    model = Mirage(weights=args.weights)
    if args.verb == "extract":
        with open(args.form, encoding="utf-8") as source:
            options["form"] = json.load(source)
    print(json.dumps(model.describe(args.image, options=options, language=args.language,
                                    keywords=args.keywords), ensure_ascii=False, indent=2))
