import getpass
import sys

from .common import print_color, mask_key, CYAN, GREEN, NC, YELLOW


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

    if not sys.stdin.isatty():
        print_color(YELLOW, "stdin is not a TTY; refusing interactive key entry. Set CACTUS_CLOUD_KEY env var instead.")
        return 0

    print()
    print(f"Get your cloud key at {CYAN}https://www.cactuscompute.com/dashboard/api-keys{NC}")
    new_key = getpass.getpass("Enter new API key (press Enter to skip): ").strip()
    if new_key:
        config.set_api_key(new_key)
        print_color(GREEN, f"API key saved: {mask_key(new_key)}")
    return 0
