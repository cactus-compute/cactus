from cactus.convert.cli import build_parser


def test_convert_parser_accepts_public_forwarded_auth_and_cache_flags():
    args = build_parser().parse_args(
        [
            "convert",
            "--model",
            "example/model",
            "--out",
            "/tmp/out",
            "--bits",
            "4",
            "--token",
            "hf_token",
            "--cache-dir",
            "/tmp/hf-cache",
        ]
    )

    assert args.token == "hf_token"
    assert args.cache_dir == "/tmp/hf-cache"
