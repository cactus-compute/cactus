from .common import print_color, GREEN


def cmd_auth(args):
    """Manage Cactus Cloud API key."""
    from .config_utils import CactusConfig

    config = CactusConfig()

    if args.clear:
        config.clear_api_key()
        print_color(GREEN, "API key cleared.")
        return 0

    api_key = config.get_api_key()

    if api_key:
        masked = api_key[:4] + "..." + api_key[-4:]
        print(f"Current API key: {masked}")
    else:
        print("No API key set.")

    if args.status:
        return 0

    print()
    print("Get your cloud key at \033[1;36mhttps://www.cactuscompute.com/dashboard/api-keys\033[0m")
    new_key = input("Enter new API key (press Enter to skip): ").strip()
    if new_key:
        config.set_api_key(new_key)
        masked = new_key[:4] + "..." + new_key[-4:]
        print_color(GREEN, f"API key saved: {masked}")
    return 0
