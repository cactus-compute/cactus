from .common import print_color, mask_key, GREEN


def cmd_auth(args):
    from .config_utils import CactusConfig

    config = CactusConfig()

    if args.clear:
        config.clear_api_key()
        print_color(GREEN, "API key cleared.")
        return 0

    api_key = config.get_api_key()

    if api_key:
        print(f"Current API key: {mask_key(api_key)}")
    else:
        print("No API key set.")

    if args.status:
        return 0

    print()
    print("Get your cloud key at \033[1;36mhttps://www.cactuscompute.com/dashboard/api-keys\033[0m")
    new_key = input("Enter new API key (press Enter to skip): ").strip()
    if new_key:
        config.set_api_key(new_key)
        print_color(GREEN, f"API key saved: {mask_key(new_key)}")
    return 0
