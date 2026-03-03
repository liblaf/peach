def is_available() -> bool:
    try:
        import cupy

        return cupy.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False
