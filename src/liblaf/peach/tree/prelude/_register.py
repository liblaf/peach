def register_pytree_prelude() -> None:
    from ._method import register_pytree_method
    from ._wrapt_function_wrapper import register_pytree_wrapt_function_wrapper
    from ._wrapt_partial import register_pytree_wrapt_partial

    register_pytree_method()
    register_pytree_wrapt_function_wrapper()
    register_pytree_wrapt_partial()
