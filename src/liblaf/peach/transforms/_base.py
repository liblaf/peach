import jarp


@jarp.define
class Transform[I, O]:
    def forward_primals(self, primals_in: I) -> O:
        raise NotImplementedError

    def forward_tangents(self, tangents_in: I) -> O:
        raise NotImplementedError

    def backward_primals(self, primals_out: O) -> I:
        raise NotImplementedError

    def backward_tangents(self, tangents_out: O) -> I:
        raise NotImplementedError

    def backward_hess_diag(self, hess_diag_out: O) -> I:
        raise NotImplementedError
