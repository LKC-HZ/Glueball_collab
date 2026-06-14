program color_singlet_reduced_projection
  use iso_fortran_env, only: real64, int64
  implicit none
  integer, parameter :: dp = real64
  complex(dp), parameter :: czero = (0.0_dp, 0.0_dp), cone = (1.0_dp, 0.0_dp), &
       cimag = (0.0_dp, 1.0_dp)
  real(dp), parameter :: sqrt3 = 1.7320508075688772_dp, half = 0.5_dp
  real(dp), parameter :: inv_sqrt2 = 0.7071067811865475_dp  ! 1/sqrt(2)
  integer :: n, m, k, npart, i, j, a
  integer :: num_args, nstates, nullity, nev_conv, nsing
  character(len=100) :: arg1, arg2, arg3
  integer, allocatable :: dims(:), stride(:), colors(:)
  complex(dp), allocatable :: T_q(:,:,:), T_aq(:,:,:), T_g(:,:,:)
  complex(dp), allocatable :: nullspace_basis(:,:)
  real(dp), parameter :: tol = 1.0e-10_dp, eps = 1.0e-14_dp
  character(len=512) :: out_line
  character(len=32)  :: tmp_str

  logical, parameter :: realify = .true.

  integer :: p, c, c2, gvar, g2, idx2, kk, sub_idx, full_idx, mapped_idx
  real(dp)    :: phase_ang
  ! For degenerate subspace separation (nullity > 1)
  complex(dp), allocatable :: tmpv(:), tmp_basis(:,:), psi_M(:,:)
  real(dp), allocatable :: rmat(:,:), aeigval(:), aeigvec(:,:)
  real(dp) :: sum_re, sum_im
  integer :: p1_swap, p2_swap, basis_cnt, cand, n_cand
  real(dp) :: proj_val, norm_val
  complex(dp), allocatable :: gm_basis(:,:), gm_mved(:), cw_basis(:,:)
  complex(dp), allocatable :: gm_psi(:,:), gm_tmpv(:)

  integer :: dim_red
  integer, allocatable :: full_to_red(:)
  integer, allocatable :: red_to_full(:)
  ! M-metric permutation for CW→GM transpose‑overlap (degenerate singlet separation)
  ! M = U U^T  swaps positive/negative weight gluon states
  integer, parameter :: M_perm(8) = [1, 2, 4, 3, 6, 5, 8, 7]
  integer, allocatable :: zw_mapped_index(:)  ! maps reduced index → reduced index under M⊗M⊗...⊗M

  integer, parameter :: nev_max_abs = 200
  integer :: nev, ncv, ldv, lworkl, ido, ar_info
  complex(dp), allocatable :: v(:,:), workd(:), workl(:), resid(:)
  real(dp), allocatable :: rwork(:)
  integer :: iparam(11), ipntr(14)
  complex(dp), allocatable :: d(:), z_red(:,:), workev(:)
  logical, allocatable :: select(:)
  real(dp) :: sigma
  character(1) :: bmat
  character(2) :: which
  external znaupd, zneupd

  complex(dp), allocatable :: temp_full1(:), temp_full2(:)

  integer :: max_states = 500000

  ! ---------- Cartan‑Weyl basis for gluons ----------
  ! Transformation matrix U_cw(i,g): CW state i = sum_g U_cw(i,g) * GM state g
  complex(dp) :: U_cw(8,8)
  ! Quantum numbers for each CW gluon state: (I3, T8)
  real(dp) :: gluon_I3(8), gluon_T8(8)
  character(len=16) :: gluon_label(8)

  ! ---------- read input ----------
  num_args = command_argument_count()
  if (num_args == 3) then
     call get_command_argument(1, arg1); read(arg1, *) n
     call get_command_argument(2, arg2); read(arg2, *) m
     call get_command_argument(3, arg3); read(arg3, *) k
  else if (num_args == 0) then
     print *, 'Enter n (quarks), m (antiquarks), k (gluons):'
     read(*, *) n, m, k
  else
     print *, 'Usage: ./color_wf n m k'; stop
  end if

  if (n < 0 .or. m < 0 .or. k < 0 .or. mod(n-m,3) /= 0) stop 'Invalid input or triality'

  nsing = count_singlets(n, m, k)
  print *, 'Theoretical singlets: ', nsing
  if (nsing == 0) stop 'No singlets'

  ! --- build Cartan‑Weyl transformation and quantum numbers ---
  call build_cartan_weyl_basis(U_cw, gluon_I3, gluon_T8, gluon_label)

  ! generators — first build Gell‑Mann, then convert gluon sector to CW
  allocate(T_q(3,3,8), T_aq(3,3,8), T_g(8,8,8))
  call init_generators_GM(T_q, T_aq, T_g)
  call transform_gluon_to_CW(T_g, U_cw)

  npart = n + m + k
  allocate(dims(npart), stride(0:npart))
  do i = 1, n; dims(i) = 3; end do
  do i = n+1, n+m; dims(i) = 3; end do
  do i = n+m+1, npart; dims(i) = 8; end do
  stride(0) = 1
  do i = 1, npart; stride(i) = stride(i-1) * dims(i); end do
  nstates = stride(npart)
  if (nstates > max_states) stop 'State space too large'
  print *, 'Full dimension:', nstates
  allocate(colors(npart))

  ! zero‑weight subspace via Cartan test (T^3 and T^8 annihilation)
  print *, 'Building zero‑weight subspace (Cartan test) ...'
  call build_zero_weight_subspace_cartan()
  print *, 'Zero‑weight dimension:', dim_red
  if (dim_red == 0) stop 'No zero‑weight states'

  allocate(temp_full1(nstates), temp_full2(nstates))

  nev = min(nev_max_abs, max(3, nsing + 10))
  nev = min(nev, dim_red - 2)
  ncv = min(max(nev+2, 5), dim_red)
  ldv = dim_red
  lworkl = ncv*(3*ncv+5) + 10
  allocate(v(ldv, ncv), workd(3*dim_red), workl(lworkl), resid(dim_red), rwork(ncv))
  allocate(d(nev), z_red(dim_red, nev), select(ncv), workev(2*ncv))

  iparam = 0
  iparam(1) = 1; iparam(3) = 5000; iparam(7) = 1
  ido = 0; ar_info = 0
  bmat = 'I'; which = 'SM'

  print *, 'ARPACK requesting', nev, ' smallest eigenvalues on reduced space ...'

  do
     call znaupd(ido, bmat, dim_red, which, nev, tol, resid, ncv, v, ldv, &
          iparam, ipntr, workd, workl, lworkl, rwork, ar_info)
     if (ido == 1 .or. ido == -1) then
        call c2_matvec_red(dim_red, workd(ipntr(1):ipntr(1)+dim_red-1), &
                           workd(ipntr(2):ipntr(2)+dim_red-1))
     else if (ido == 99) then
        exit
     else
        print *, 'znaupd error, info =', ar_info; stop
     end if
  end do
  if (ar_info /= 0) stop 'znaupd failed'

  select = .true.; sigma = 0.0_dp
  call zneupd(.true., 'A', select, d, z_red, dim_red, sigma, workev, bmat, &
       dim_red, which, nev, tol, resid, ncv, v, ldv, &
       iparam, ipntr, workd, workl, lworkl, rwork, ar_info)
  if (ar_info /= 0) stop 'zneupd failed'

  nev_conv = iparam(5)
  print *, 'Converged eigenvalues: ', nev_conv

  nullity = 0
  do i = 1, nev_conv
     if (abs(d(i)) < tol) nullity = nullity + 1
  end do
  print *, 'Actual number of singlets: ', nullity

  if (nullity /= nsing) then
     if (nullity < nsing) print *, 'WARNING: fewer singlets found'
     if (nullity > nsing) stop 'ERROR: too many singlets'
  end if
  if (nullity == 0) stop 'No singlet states'

  ! map reduced eigenvectors to full space
  allocate(nullspace_basis(nstates, nullity))
  nullspace_basis = czero
  kk = 0
  do i = 1, nev_conv
     if (abs(d(i)) < tol) then
        kk = kk + 1
        do j = 1, dim_red
           nullspace_basis(red_to_full(j)+1, kk) = z_red(j, i)
        end do
     end if
  end do

  ! realification — get purely real CW coefficients for all singlets
  ! Algorithm: phase‑fix in CW basis → Gram‑Schmidt → M‑diagonalisation (CW)
  !   Phase φ = ½·arg(Σ ψᵢ²) maximises real CW content.
  !   M diagonalisation labels d‑type (M=+1) / f‑type (M=−1) without phase change.
  !   Then P₁₂ diagonalisation for d/f labelling
  if (realify) then
     allocate(tmpv(nstates), tmp_basis(nstates, nullity))
     tmp_basis(:, 1:nullity) = nullspace_basis(:, 1:nullity)

     ! ---- Step 1: Phase‑fix each singlet to maximise real CW coefficients ----
     ! Colour singlets have inherently real CW coefficients (up to a global
     ! phase from ARPACK).  Optimal phase: φ = ½·arg(Σᵢ ψᵢ²), then ψ → e⁻ⁱᶠ ψ.
     do j = 1, nullity
        phase_ang = 0.5_dp * atan2(aimag(sum(tmp_basis(:, j)**2)), &
                                     real(sum(tmp_basis(:, j)**2), dp))
        tmp_basis(:, j) = tmp_basis(:, j) * cmplx(cos(-phase_ang), sin(-phase_ang), dp)
     end do

     ! ---- Step 2: Zero out residual imaginary parts (numerical noise) ----
     do j = 1, nullity
        do i = 1, nstates
           tmp_basis(i, j) = cmplx(real(tmp_basis(i, j), dp), 0.0_dp, dp)
        end do
     end do

     ! ---- Step 3: Real Gram‑Schmidt in CW basis ----
     kk = 0
     do cand = 1, nullity
        tmpv = tmp_basis(:, cand)
        do i = 1, kk
           proj_val = real(sum(conjg(tmp_basis(:, i)) * tmpv), dp)
           tmpv = tmpv - proj_val * tmp_basis(:, i)
        end do
        norm_val = sqrt(real(sum(conjg(tmpv) * tmpv), dp))
        if (norm_val > tol) then
           kk = kk + 1
           tmp_basis(:, kk) = tmpv / norm_val
        end if
     end do
     if (kk /= nullity) then
        print *, 'ERROR: CW Gram‑Schmidt rank:', kk, '≠', nullity; stop
     end if

     ! ---- Step 4: M‑diagonalisation in CW basis (only when n == m, nullity > 1) ----
     ! M inverts all weights: swaps quark↔antiquark colours + gluon M_perm.
     ! This only preserves the zero‑weight subspace when n == m.
     ! Note: no ×i phase rotation here — CW phase‑fix already ensures real CW
     ! coefficients for all singlets.  Multiplying M=−1 eigenvectors by i would
     ! flip the convention (CW imaginary, GM real) and break pure‑gluon output.
     if (n == m .and. nullity > 1) then
        allocate(rmat(nullity, nullity), aeigval(nullity), aeigvec(nullity, nullity))
        allocate(psi_M(nstates, nullity))
        psi_M = tmp_basis
        do j = 1, nullity
           call apply_M_full(psi_M(:, j), tmpv)
           do i = 1, nullity
              rmat(i, j) = real(sum(conjg(psi_M(:, i)) * tmpv), dp)
           end do
        end do
        call jacobi_diag(nullity, rmat, aeigval, aeigvec)
        do kk = 1, nullity
           tmp_basis(:, kk) = czero
           do j = 1, nullity
              tmp_basis(:, kk) = tmp_basis(:, kk) + aeigvec(j, kk) * psi_M(:, j)
           end do
        end do
        deallocate(psi_M, rmat, aeigvec, aeigval)
     end if
     ! When n ≠ m or nullity == 1: no M‑diagonalisation needed

     ! ---- Step 5: P₁₂ interchange for d/f labelling ----
     p1_swap = 0; p2_swap = 0
     if (k >= 2) then
        p1_swap = n + m + 1; p2_swap = n + m + 2
     else if (n >= 2) then
        p1_swap = 1; p2_swap = 2
     else if (m >= 2) then
        p1_swap = n + 1; p2_swap = n + 2
     end if

     if (p1_swap > 0) then
        allocate(rmat(nullity, nullity), aeigval(nullity), aeigvec(nullity, nullity))
        do j = 1, nullity
           do kk = 1, nullity
              rmat(j, kk) = swap_overlap(tmp_basis(:, j), tmp_basis(:, kk), p1_swap, p2_swap)
           end do
        end do
        call jacobi_diag(nullity, rmat, aeigval, aeigvec)
        allocate(psi_M(nstates, nullity))
        psi_M = tmp_basis
        do kk = 1, nullity
           tmp_basis(:, kk) = czero
           do j = 1, nullity
              tmp_basis(:, kk) = tmp_basis(:, kk) + aeigvec(j, kk) * psi_M(:, j)
           end do
        end do
        deallocate(psi_M)
        ! Sort: +1 (d‑type) first, −1 (f‑type) last
        do i = 1, nullity - 1
           kk = i
           do j = i + 1, nullity
              if (aeigval(j) > aeigval(kk)) kk = j
           end do
           if (kk /= i) then
              sum_re = aeigval(i); aeigval(i) = aeigval(kk); aeigval(kk) = sum_re
              tmpv = tmp_basis(:, i)
              tmp_basis(:, i) = tmp_basis(:, kk)
              tmp_basis(:, kk) = tmpv
           end if
        end do
        deallocate(rmat, aeigvec)
        ! Keep aeigval allocated for output labels (CW + GM sections)
     else
        ! No pair available for interchange test — aeigval not allocated
        ! Output will use sum_re/sum_im fallback for labelling
     end if

     ! Copy result back to nullspace_basis
     nullspace_basis(:, 1:nullity) = tmp_basis(:, 1:nullity)

     ! ---- Final: normalise and clean numerical noise ----
     do j = 1, nullity
        nullspace_basis(:, j) = nullspace_basis(:, j) / sqrt(sum(abs(nullspace_basis(:, j))**2))
        do i = 1, nstates
           if (abs(aimag(nullspace_basis(i, j))) < 1d-10 * max(tol, abs(real(nullspace_basis(i, j), dp)))) &
                nullspace_basis(i, j) = cmplx(real(nullspace_basis(i, j), dp), 0.0_dp, dp)
        end do
     end do

     ! ---- Step 6: Reconstruct GM basis for output ----
     allocate(gm_psi(nstates, nullity), gm_tmpv(nstates))
     do j = 1, nullity
        call cw_to_gm_vec(nullspace_basis(:, j), gm_tmpv, temp_full1, temp_full2)
        gm_psi(:, j) = gm_tmpv
     end do
     deallocate(gm_tmpv)

     deallocate(tmpv, tmp_basis)
  end if

  ! output wavefunctions — with Cartan‑Weyl labels for gluons
  do j = 1, nullity
     sum_re = sum(real(nullspace_basis(:, j), dp)**2)
     sum_im = sum(aimag(nullspace_basis(:, j))**2)
     ! In degenerate case, aeigval(j) carries interchange eigenvalue: +1 → d‑type, −1 → f‑type
     if (allocated(aeigval)) then
        if (aeigval(j) >= 0.0_dp) then
           print *, '--- Singlet ', j, ' (real; d‑type) ---'
        else
           print *, '--- Singlet ', j, ' (real; f‑type) ---'
        end if
     else
        if (sum_re >= sum_im) then
           print *, '--- Singlet ', j, ' (real) ---'
        else
           print *, '--- Singlet ', j, ' (imag) ---'
        end if
     end if
     do i = 0, nstates-1
        if (abs(nullspace_basis(i+1, j)) > tol) then
           call decode(i, colors, npart, dims)
           out_line = ''
           do p = 1, n
              write(tmp_str, '(I1)') colors(p)+1
              out_line = trim(out_line) // 'q' // tmp_str
              if (p < n) out_line = trim(out_line) // ','
           end do
           if (n > 0 .and. (m+k) > 0) out_line = trim(out_line) // ' | '
           do p = n+1, n+m
              write(tmp_str, '(I1)') colors(p)+1
              out_line = trim(out_line) // 'qb' // tmp_str
              if (p < n+m) out_line = trim(out_line) // ','
           end do
           if (m > 0 .and. k > 0) out_line = trim(out_line) // ' | '
           do p = n+m+1, npart
              ! CW label with (I3, T8) quantum numbers
              out_line = trim(out_line) // trim(gluon_label(colors(p)+1))
              if (p < npart) out_line = trim(out_line) // ','
           end do
           write(*, '(A, SP, F12.8, SP, F12.8, "i")') trim(out_line)//' : ', &
                nullspace_basis(i+1, j)
        end if
     end do
  end do

  ! ---- Also output in Gell‑Mann basis ----
  if (allocated(gm_psi)) then
     print *, ''
     print *, '===== Gell-Mann basis representation ====='
     do j = 1, nullity
        ! Label
        if (allocated(aeigval)) then
           if (aeigval(j) >= 0.0_dp) then
              print *, '--- Singlet ', j, ' (GM; d-type) ---'
           else
              print *, '--- Singlet ', j, ' (GM; f-type) ---'
           end if
        else
           print *, '--- Singlet ', j, ' (GM) ---'
        end if
        do i = 0, nstates-1
           if (abs(gm_psi(i+1, j)) > tol) then
              call decode(i, colors, npart, dims)
              out_line = ''
              do p = 1, n
                 write(tmp_str, '(I1)') colors(p)+1
                 out_line = trim(out_line) // 'q' // tmp_str
                 if (p < n) out_line = trim(out_line) // ','
              end do
              if (n > 0 .and. (m+k) > 0) out_line = trim(out_line) // ' | '
              do p = n+1, n+m
                 write(tmp_str, '(I1)') colors(p)+1
                 out_line = trim(out_line) // 'qb' // tmp_str
                 if (p < n+m) out_line = trim(out_line) // ','
              end do
              if (m > 0 .and. k > 0) out_line = trim(out_line) // ' | '
              do p = n+m+1, npart
                 write(tmp_str, '(I1)') colors(p)+1
                 out_line = trim(out_line) // 'lam' // trim(tmp_str)
                 if (p < npart) out_line = trim(out_line) // ','
              end do
              write(*, '(A, SP, F12.8, SP, F12.8, "i")') trim(out_line)//' : ', &
                   gm_psi(i+1, j)
           end if
        end do
     end do
  end if

  stop

contains

  ! ---------------------------------------------------------------
  ! Build Cartan‑Weyl transformation matrix and gluon quantum numbers
  ! ---------------------------------------------------------------
  subroutine build_cartan_weyl_basis(U, I3_arr, T8_arr, labels)
    complex(dp), intent(out) :: U(8,8)
    real(dp),    intent(out) :: I3_arr(8), T8_arr(8)
    character(len=16), intent(out) :: labels(8)

    U = czero

    ! CW 1 = |3⟩_GM   Cartan H1:  I3=0, T8=0
    U(1,3) = cone
    I3_arr(1) = 0.0_dp;  T8_arr(1) = 0.0_dp
    labels(1) = 'g_C1'

    ! CW 2 = |8⟩_GM   Cartan H2:  I3=0, T8=0
    U(2,8) = cone
    I3_arr(2) = 0.0_dp;  T8_arr(2) = 0.0_dp
    labels(2) = 'g_C2'

    ! CW 3 = (|1⟩+i|2⟩)/√2   root:  I3=+1, T8=0
    U(3,1) =  inv_sqrt2 * cone
    U(3,2) =  inv_sqrt2 * cimag
    I3_arr(3) = 1.0_dp;   T8_arr(3) = 0.0_dp
    labels(3) = 'g(+1,0)'

    ! CW 4 = (|1⟩-i|2⟩)/√2   root:  I3=-1, T8=0
    U(4,1) =  inv_sqrt2 * cone
    U(4,2) = -inv_sqrt2 * cimag
    I3_arr(4) = -1.0_dp;  T8_arr(4) = 0.0_dp
    labels(4) = 'g(-1,0)'

    ! CW 5 = (|4⟩+i|5⟩)/√2   root:  I3=+1/2, T8=+√3/2
    U(5,4) =  inv_sqrt2 * cone
    U(5,5) =  inv_sqrt2 * cimag
    I3_arr(5) = 0.5_dp;   T8_arr(5) = 0.5_dp * sqrt3
    labels(5) = 'g(+1/2,+s3/2)'

    ! CW 6 = (|4⟩-i|5⟩)/√2   root:  I3=-1/2, T8=-√3/2
    U(6,4) =  inv_sqrt2 * cone
    U(6,5) = -inv_sqrt2 * cimag
    I3_arr(6) = -0.5_dp;  T8_arr(6) = -0.5_dp * sqrt3
    labels(6) = 'g(-1/2,-s3/2)'

    ! CW 7 = (|6⟩+i|7⟩)/√2   root:  I3=-1/2, T8=+√3/2
    U(7,6) =  inv_sqrt2 * cone
    U(7,7) =  inv_sqrt2 * cimag
    I3_arr(7) = -0.5_dp;  T8_arr(7) = 0.5_dp * sqrt3
    labels(7) = 'g(-1/2,+s3/2)'

    ! CW 8 = (|6⟩-i|7⟩)/√2   root:  I3=+1/2, T8=-√3/2
    U(8,6) =  inv_sqrt2 * cone
    U(8,7) = -inv_sqrt2 * cimag
    I3_arr(8) = 0.5_dp;   T8_arr(8) = -0.5_dp * sqrt3
    labels(8) = 'g(+1/2,-s3/2)'
  end subroutine build_cartan_weyl_basis

  ! ---------------------------------------------------------------
  ! Transform gluon generators from Gell‑Mann to Cartan‑Weyl basis
  !   |i_CW> = sum_g U(i,g) |g_GM>   where U(i,g) = <g_GM | i_CW>
  !   Operator transform: T^{CW} = U* @ T^{GM} @ U^T
  !   T_g^{CW}(i,j,a) = sum_{b,c} conjg(U(i,b)) * T_g^{GM}(b,c,a) * U(j,c)
  ! ---------------------------------------------------------------
  subroutine transform_gluon_to_CW(Tg, U)
    complex(dp), intent(inout) :: Tg(8,8,8)
    complex(dp), intent(in)    :: U(8,8)
    complex(dp) :: Tg_GM(8,8,8), temp(8,8)
    integer :: a, i, j, b, c

    ! save GM copy
    Tg_GM = Tg

    do a = 1, 8
       ! temp(b,j) = sum_c Tg_GM(b,c,a) * U(j,c)
       temp = czero
       do j = 1, 8
          do b = 1, 8
             do c = 1, 8
                temp(b,j) = temp(b,j) + Tg_GM(b,c,a) * U(j,c)
             end do
          end do
       end do
       ! Tg(i,j,a) = sum_b conjg(U(i,b)) * temp(b,j)
       do j = 1, 8
          do i = 1, 8
             Tg(i,j,a) = czero
             do b = 1, 8
                Tg(i,j,a) = Tg(i,j,a) + conjg(U(i,b)) * temp(b,j)
             end do
          end do
       end do
    end do
  end subroutine transform_gluon_to_CW

  ! ---------------------------------------------------------------
  ! Zero‑weight test using Cartan generators T^3 and T^8
  ! ---------------------------------------------------------------
  subroutine build_zero_weight_subspace_cartan()
    integer :: idx, cnt, num3, num8, list3(200), list8(200)
    complex(dp) :: coeff3(200), coeff8(200)
    integer :: mapped_idx, mapped_labels(npart), sub_idx, p, g
    allocate(full_to_red(0:nstates-1))
    full_to_red = 0
    dim_red = 0
    do idx = 0, nstates-1
       call t_action(3, idx, num3, list3, coeff3)
       call t_action(8, idx, num8, list8, coeff8)
       if (num3 == 0 .and. num8 == 0) then
          dim_red = dim_red + 1
          full_to_red(idx) = dim_red
       end if
    end do
    allocate(red_to_full(dim_red), zw_mapped_index(dim_red))
    cnt = 0
    do idx = 0, nstates-1
       if (full_to_red(idx) > 0) then
          cnt = cnt + 1
          red_to_full(cnt) = idx
       end if
    end do

    ! Build M-metric mapping: M inverts ALL weights.
    ! For gluons: M swaps positive/negative weight states (M_perm).
    ! For quarks↔antiquarks (when n=m): M swaps color indices of paired
    !   quark i and antiquark i, since -weight(q_c) = weight(qb_c).
    ! M preserves the zero-weight subspace only when n == m.
    do sub_idx = 1, dim_red
       idx = red_to_full(sub_idx)
       call decode(idx, colors, npart, dims)
       mapped_labels = colors
       ! Swap quark ↔ antiquark color indices (pairs position i ↔ position n+i)
       do p = 1, min(n, m)
          mapped_labels(p) = colors(n + p)
          mapped_labels(n + p) = colors(p)
       end do
       ! Apply M_perm to each gluon slot
       do p = n+m+1, npart
          g = colors(p) + 1                     ! 1-based CW gluon index
          mapped_labels(p) = M_perm(g) - 1       ! back to 0-based
       end do
       ! Encode back
       mapped_idx = 0
       do p = 1, npart
          mapped_idx = mapped_idx + mapped_labels(p) * stride(p-1)
       end do
       ! Look up reduced index (must be in zero-weight subspace when n==m)
       zw_mapped_index(sub_idx) = full_to_red(mapped_idx)
       if (zw_mapped_index(sub_idx) <= 0 .and. n == m) then
          print *, 'ERROR: M-metric mapping left zero-weight subspace'
          print *, '  sub_idx=', sub_idx, ' mapped_idx=', mapped_idx
          print *, '  original colors=', colors
          print *, '  mapped   colors=', mapped_labels
          stop
       end if
    end do
  end subroutine build_zero_weight_subspace_cartan

  ! ---------------------------------------------------------------
  ! action of total generator T^a on a single basis state
  ! ---------------------------------------------------------------
  subroutine t_action(a, global_idx, num, idx_list, coeff_list)
    integer, intent(in) :: a, global_idx
    integer, intent(out) :: num, idx_list(:)
    complex(dp), intent(out) :: coeff_list(:)
    integer :: p, c, c2, gvar, g2, idx2
    complex(dp) :: diag, val
    num = 0; diag = czero
    call decode(global_idx, colors, npart, dims)
    do p = 1, n
       c = colors(p)+1; diag = diag + T_q(c,c,a)
    end do
    do p = n+1, n+m
       c = colors(p)+1; diag = diag + T_aq(c,c,a)
    end do
    do p = n+m+1, npart
       gvar = colors(p)+1; diag = diag + T_g(gvar,gvar,a)
    end do
    if (abs(diag) > tol) then
       num = num+1; idx_list(num) = global_idx; coeff_list(num) = diag
    end if
    do p = 1, n
       c = colors(p)+1
       do c2 = 1, 3
          if (c2 == c) cycle
          val = T_q(c2,c,a)
          if (abs(val) > tol) then
             idx2 = global_idx + (c2-1 - (c-1)) * stride(p-1)
             num = num+1; idx_list(num) = idx2; coeff_list(num) = val
          end if
       end do
    end do
    do p = n+1, n+m
       c = colors(p)+1
       do c2 = 1, 3
          if (c2 == c) cycle
          val = T_aq(c2,c,a)
          if (abs(val) > tol) then
             idx2 = global_idx + (c2-1 - (c-1)) * stride(p-1)
             num = num+1; idx_list(num) = idx2; coeff_list(num) = val
          end if
       end do
    end do
    do p = n+m+1, npart
       gvar = colors(p)+1
       do g2 = 1, 8
          if (g2 == gvar) cycle
          val = T_g(g2,gvar,a)
          if (abs(val) > tol) then
             idx2 = global_idx + (g2-1 - (gvar-1)) * stride(p-1)
             num = num+1; idx_list(num) = idx2; coeff_list(num) = val
          end if
       end do
    end do
  end subroutine t_action

  ! ---------------------------------------------------------------
  ! reduced matrix‑vector product via prolongation/restriction
  ! ---------------------------------------------------------------
  subroutine c2_matvec_red(nr, x_red, y_red)
    integer, intent(in) :: nr
    complex(dp), intent(in)  :: x_red(nr)
    complex(dp), intent(out) :: y_red(nr)
    complex(dp), allocatable :: x_full(:), y_full(:)
    integer :: i_red, full_idx
    allocate(x_full(nstates), y_full(nstates))
    x_full = czero
    do i_red = 1, nr
       full_idx = red_to_full(i_red) + 1
       x_full(full_idx) = x_red(i_red)
    end do
    call c2_matvec_full(nstates, x_full, y_full)
    do i_red = 1, nr
       full_idx = red_to_full(i_red) + 1
       y_red(i_red) = y_full(full_idx)
    end do
    deallocate(x_full, y_full)
  end subroutine c2_matvec_red

  ! full space C2 matrix‑vector product
  subroutine c2_matvec_full(nst, x, y)
    integer, intent(in) :: nst
    complex(dp), intent(in)  :: x(nst)
    complex(dp), intent(out) :: y(nst)
    integer :: a
    complex(dp), allocatable :: t1(:), t2(:)
    y = czero
    do a = 1, 8
       allocate(t1(nst), t2(nst))
       t1 = czero
       call apply_Ta(a, nst, x, t1)
       t2 = czero
       call apply_Ta(a, nst, t1, t2)
       y = y + t2
       deallocate(t1, t2)
    end do
  end subroutine c2_matvec_full

  subroutine apply_Ta(a, nst, in, out)
    integer, intent(in) :: a, nst
    complex(dp), intent(in) :: in(nst)
    complex(dp), intent(inout) :: out(nst)
    integer :: idx, p, c, c2, gvar, g2, idx2
    complex(dp) :: diag, val
    do idx = 0, nst-1
       if (abs(in(idx+1)) < 1.0e-15_dp) cycle
       call decode(idx, colors, npart, dims)
       diag = czero
       do p = 1, n; c = colors(p)+1; diag = diag + T_q(c,c,a); end do
       do p = n+1, n+m; c = colors(p)+1; diag = diag + T_aq(c,c,a); end do
       do p = n+m+1, npart; gvar = colors(p)+1; diag = diag + T_g(gvar,gvar,a); end do
       out(idx+1) = out(idx+1) + diag * in(idx+1)
       do p = 1, n
          c = colors(p)+1
          do c2 = 1, 3; if (c2 == c) cycle
             val = T_q(c2,c,a)
             if (abs(val) > tol) then
                idx2 = idx + (c2-1-(c-1))*stride(p-1)
                out(idx2+1) = out(idx2+1) + val * in(idx+1)
             end if
          end do
       end do
       do p = n+1, n+m
          c = colors(p)+1
          do c2 = 1, 3; if (c2 == c) cycle
             val = T_aq(c2,c,a)
             if (abs(val) > tol) then
                idx2 = idx + (c2-1-(c-1))*stride(p-1)
                out(idx2+1) = out(idx2+1) + val * in(idx+1)
             end if
          end do
       end do
       do p = n+m+1, npart
          gvar = colors(p)+1
          do g2 = 1, 8; if (g2 == gvar) cycle
             val = T_g(g2,gvar,a)
             if (abs(val) > tol) then
                idx2 = idx + (g2-1-(gvar-1))*stride(p-1)
                out(idx2+1) = out(idx2+1) + val * in(idx+1)
             end if
          end do
       end do
    end do
  end subroutine apply_Ta

  ! ----- auxiliary functions -----
  subroutine decode(idx, col, np, dim_arr)
    integer, intent(in) :: idx, np, dim_arr(np)
    integer, intent(out) :: col(np)
    integer :: temp, p
    temp = idx
    do p = 1, np; col(p) = mod(temp, dim_arr(p)); temp = temp/dim_arr(p); end do
  end subroutine

  function count_singlets(n, m, k) result(num)
    integer, intent(in) :: n, m, k
    integer(int64) :: num
    integer :: max_dim, i
    integer(int64), allocatable :: multi(:,:), t1(:,:), t2(:,:)
    if (mod(n - m, 3) /= 0) then; num = 0; return; end if
    max_dim = n + m + 2*k + 5
    allocate(multi(0:max_dim, 0:max_dim), t1(0:max_dim, 0:max_dim), t2(0:max_dim, 0:max_dim))
    multi = 0_int64; multi(0,0) = 1_int64
    do i = 1, n
       t1 = 0_int64; call multiply_quark(multi, max_dim, t1); multi = t1
    end do
    do i = 1, m
       t1 = 0_int64; call multiply_antiquark(multi, max_dim, t1); multi = t1
    end do
    do i = 1, k
       t1 = 0_int64; call multiply_gluon(multi, max_dim, t1, t2); multi = t1
    end do
    num = multi(0,0)
    deallocate(multi, t1, t2)
  end function

  subroutine multiply_quark(A, dim, B)
    integer(int64), intent(in) :: A(0:, 0:)
    integer, intent(in) :: dim
    integer(int64), intent(out) :: B(0:, 0:)
    integer :: p, q
    integer(int64) :: c
    B = 0_int64
    do p = 0, dim
       do q = 0, dim
          c = A(p,q)
          if (c == 0) cycle
          if (p+1 <= dim) B(p+1, q) = B(p+1, q) + c
          if (p > 0 .and. q+1 <= dim) B(p-1, q+1) = B(p-1, q+1) + c
          if (q > 0) B(p, q-1) = B(p, q-1) + c
       end do
    end do
  end subroutine multiply_quark

  subroutine multiply_antiquark(A, dim, B)
    integer(int64), intent(in) :: A(0:, 0:)
    integer, intent(in) :: dim
    integer(int64), intent(out) :: B(0:, 0:)
    integer :: p, q
    integer(int64) :: c
    B = 0_int64
    do p = 0, dim
       do q = 0, dim
          c = A(p,q)
          if (c == 0) cycle
          if (q+1 <= dim) B(p, q+1) = B(p, q+1) + c
          if (q > 0 .and. p+1 <= dim) B(p+1, q-1) = B(p+1, q-1) + c
          if (p > 0) B(p-1, q) = B(p-1, q) + c
       end do
    end do
  end subroutine multiply_antiquark

  subroutine multiply_gluon(A, dim, B, T)
    integer(int64), intent(in) :: A(0:, 0:)
    integer, intent(in) :: dim
    integer(int64), intent(out) :: B(0:, 0:), T(0:, 0:)
    integer :: p, q
    integer(int64) :: c
    T = 0_int64
    do p = 0, dim
       do q = 0, dim
          c = A(p,q)
          if (c == 0) cycle
          if (p+1 <= dim) T(p+1, q) = T(p+1, q) + c
          if (p > 0 .and. q+1 <= dim) T(p-1, q+1) = T(p-1, q+1) + c
          if (q > 0) T(p, q-1) = T(p, q-1) + c
       end do
    end do
    B = 0_int64
    do p = 0, dim
       do q = 0, dim
          c = T(p,q)
          if (c == 0) cycle
          if (q+1 <= dim) B(p, q+1) = B(p, q+1) + c
          if (q > 0 .and. p+1 <= dim) B(p+1, q-1) = B(p+1, q-1) + c
          if (p > 0) B(p-1, q) = B(p-1, q) + c
       end do
    end do
    do p = 0, dim
       do q = 0, dim
          B(p,q) = B(p,q) - A(p,q)
       end do
    end do
  end subroutine multiply_gluon

  ! ---------------------------------------------------------------
  ! Initialize generators in Gell‑Mann basis (original routine)
  ! ---------------------------------------------------------------
  subroutine init_generators_GM(Tq, Taq, Tg)
    complex(dp), intent(out) :: Tq(3,3,8), Taq(3,3,8), Tg(8,8,8)
    complex(dp) :: lambda(3,3,8), f(8,8,8)
    integer :: a, i, j
    lambda = czero
    lambda(1,2,1) = cone; lambda(2,1,1) = cone
    lambda(1,2,2) = -cimag; lambda(2,1,2) = cimag
    lambda(1,1,3) = cone; lambda(2,2,3) = -cone
    lambda(1,3,4) = cone; lambda(3,1,4) = cone
    lambda(1,3,5) = -cimag; lambda(3,1,5) = cimag
    lambda(2,3,6) = cone; lambda(3,2,6) = cone
    lambda(2,3,7) = -cimag; lambda(3,2,7) = cimag
    lambda(1,1,8) = cone/sqrt3; lambda(2,2,8) = cone/sqrt3
    lambda(3,3,8) = -2.0_dp*cone/sqrt3
    Tq = lambda * half
    do a = 1, 8
       Taq(:,:,a) = -transpose(lambda(:,:,a)) * half
    end do
    f = czero
    f(1,2,3) = cone
    f(1,4,7) = half*cone;  f(1,5,6) = -half*cone
    f(2,4,6) = half*cone;  f(2,5,7) = half*cone
    f(3,4,5) = half*cone;  f(3,6,7) = -half*cone
    f(4,5,8) = sqrt3*half*cone
    f(6,7,8) = sqrt3*half*cone
    do i = 1,8; do j = 1,8; do a = 1,8
       if (abs(f(i,j,a)) > 0.0_dp) then
          f(j,a,i) = f(i,j,a); f(a,i,j) = f(i,j,a)
          f(j,i,a) = -f(i,j,a); f(i,a,j) = -f(i,j,a); f(a,j,i) = -f(i,j,a)
       end if
    end do; end do; end do
    do a = 1, 8; do i = 1, 8; do j = 1, 8
       Tg(i,j,a) = -cimag * f(a,i,j)
    end do; end do; end do
  end subroutine init_generators_GM

  ! ---------------------------------------------------------------
  ! Jacobi diagonalisation of real symmetric matrix (small dimension, <~10)
  ! Uses rotation J = [[c, s], [-s, c]] with tan(2θ) = 2*A(p,q)/(A(q,q)-A(p,p))
  ! ---------------------------------------------------------------
  subroutine jacobi_diag(n, A, eigval, eigvec)
    integer, intent(in) :: n
    real(dp), intent(inout) :: A(n,n)
    real(dp), intent(out) :: eigval(n), eigvec(n,n)
    integer :: i, j, p, q, iter, max_iter
    real(dp) :: theta, c, s, off, apq, app, aqq, aip, aiq, vip, viq
    real(dp), parameter :: conv_tol = 1.0e-14_dp

    max_iter = 100
    eigvec = 0.0_dp
    do i = 1, n
       eigvec(i,i) = 1.0_dp
    end do

    do iter = 1, max_iter
       ! Find largest off-diagonal element
       off = 0.0_dp; p = 1; q = 2
       do i = 1, n-1
          do j = i+1, n
             if (abs(A(i,j)) > off) then
                off = abs(A(i,j)); p = i; q = j
             end if
          end do
       end do
       if (off < conv_tol) exit

       app = A(p,p); aqq = A(q,q); apq = A(p,q)
       theta = 0.5_dp * atan2(2.0_dp * apq, aqq - app)
       c = cos(theta); s = sin(theta)

       ! Diagonal block: A' = J^T A J  with J = [[c, s], [-s, c]]
       A(p,p) = c*c*app + s*s*aqq - 2.0_dp*c*s*apq
       A(q,q) = s*s*app + c*c*aqq + 2.0_dp*c*s*apq
       A(p,q) = 0.0_dp; A(q,p) = 0.0_dp

       ! Off-diagonal rows/cols: A'(i,p) = c*A(i,p) - s*A(i,q)
       do i = 1, n
          if (i /= p .and. i /= q) then
             aip = A(i,p); aiq = A(i,q)
             A(i,p) = c*aip - s*aiq; A(p,i) = A(i,p)
             A(i,q) = s*aip + c*aiq; A(q,i) = A(i,q)
          end if
       end do

       ! Eigenvectors: V' = V J  =>  V'(i,p) = c*V(i,p) - s*V(i,q)
       do i = 1, n
          vip = eigvec(i,p); viq = eigvec(i,q)
          eigvec(i,p) = c*vip - s*viq
          eigvec(i,q) = s*vip + c*viq
       end do
    end do

    do i = 1, n
       eigval(i) = A(i,i)
    end do
  end subroutine jacobi_diag

  ! ---------------------------------------------------------------
  ! Transform full‑space vector from CW basis to GM basis
  ! For each gluon particle, apply U^T transformation:
  !   ψ_GM(g) = Σ_i U(i,g) ψ_CW(i)  for each gluon index
  ! Quark and antiquark indices are unchanged.
  ! ---------------------------------------------------------------
  subroutine cw_to_gm_vec(psi_cw, psi_gm, buf1, buf2)
    complex(dp), intent(in)    :: psi_cw(nstates)
    complex(dp), intent(out)   :: psi_gm(nstates)
    complex(dp), intent(inout) :: buf1(nstates), buf2(nstates)
    integer :: p, idx, idx2, gcw, ggm, ii
    integer :: col(npart)
    buf1 = psi_cw
    do p = n + m + 1, npart
       buf2 = czero
       do idx = 0, nstates - 1
          if (abs(buf1(idx+1)) < 1.0e-30_dp) cycle
          call decode(idx, col, npart, dims)
          gcw = col(p) + 1
          do ggm = 1, 8
             if (abs(U_cw(gcw, ggm)) < 1.0e-30_dp) cycle
             col(p) = ggm - 1
             idx2 = 0
             do ii = 1, npart
                idx2 = idx2 + col(ii) * stride(ii-1)
             end do
             buf2(idx2+1) = buf2(idx2+1) + U_cw(gcw, ggm) * buf1(idx+1)
          end do
          col(p) = gcw - 1
       end do
       buf1 = buf2
    end do
    psi_gm = buf1
  end subroutine cw_to_gm_vec

  ! ---------------------------------------------------------------
  ! Transform full‑space vector from GM basis to CW basis
  ! For each gluon particle, apply conj(U) transformation:
  !   ψ_CW(i) = Σ_g conj(U(i,g)) ψ_GM(g)  for each gluon index
  ! ---------------------------------------------------------------
  subroutine gm_to_cw_vec(psi_gm, psi_cw, buf1, buf2)
    complex(dp), intent(in)    :: psi_gm(nstates)
    complex(dp), intent(out)   :: psi_cw(nstates)
    complex(dp), intent(inout) :: buf1(nstates), buf2(nstates)
    integer :: p, idx, idx2, gcw, ggm, ii
    integer :: col(npart)
    buf1 = psi_gm
    do p = n + m + 1, npart
       buf2 = czero
       do idx = 0, nstates - 1
          if (abs(buf1(idx+1)) < 1.0e-30_dp) cycle
          call decode(idx, col, npart, dims)
          ggm = col(p) + 1
          do gcw = 1, 8
             if (abs(conjg(U_cw(gcw, ggm))) < 1.0e-30_dp) cycle
             col(p) = gcw - 1
             idx2 = 0
             do ii = 1, npart
                idx2 = idx2 + col(ii) * stride(ii-1)
             end do
             buf2(idx2+1) = buf2(idx2+1) + conjg(U_cw(gcw, ggm)) * buf1(idx+1)
          end do
          col(p) = ggm - 1
       end do
       buf1 = buf2
    end do
    psi_cw = buf1
  end subroutine gm_to_cw_vec

  ! ---------------------------------------------------------------
  ! Interchange (swap) overlap of two real GM‑basis vectors
  ! Computes ⟨P₁₂ v₁ | v₂⟩ where P₁₂ swaps particle indices p1, p2
  ! Uses full complex inner product so it works for both real and imaginary vectors.
  ! ---------------------------------------------------------------
  function swap_overlap(v1, v2, p1, p2) result(ov)
    complex(dp), intent(in) :: v1(nstates), v2(nstates)
    integer, intent(in) :: p1, p2
    real(dp) :: ov
    integer :: idx, idx2, ii, tmp_col
    integer :: col(npart)
    ov = 0.0_dp
    do idx = 0, nstates - 1
       if (abs(v2(idx+1)) < tol) cycle
       call decode(idx, col, npart, dims)
       ! Swap positions p1 and p2
       tmp_col = col(p1)
       col(p1) = col(p2)
       col(p2) = tmp_col
       idx2 = 0
       do ii = 1, npart
          idx2 = idx2 + col(ii) * stride(ii-1)
       end do
       ov = ov + real(conjg(v1(idx2+1)) * v2(idx+1), dp)
    end do
  end function swap_overlap

  ! ---------------------------------------------------------------
  ! Apply the M operator (total weight inversion) to a full-space CW vector
  ! M inverts all weights: swaps quark↔antiquark colour indices (when n=m)
  !   and applies M_perm to each gluon (positive↔negative weight flip).
  ! For quark/antiquark pairs: M maps |q_c, qb_c'> → |q_{c'}, qb_c|
  !   since -weight(q_c) = weight(qb_c).
  ! Result: (Mψ)(s) = ψ(M⁻¹(s))
  ! ---------------------------------------------------------------
  subroutine apply_M_full(psi_in, psi_out)
    complex(dp), intent(in)  :: psi_in(nstates)
    complex(dp), intent(out) :: psi_out(nstates)
    integer :: idx, idx2, ii, p, g, tmp
    integer :: col(npart), mapped_col(npart)

    psi_out = czero
    do idx = 0, nstates - 1
       if (abs(psi_in(idx+1)) < 1.0e-30_dp) cycle
       call decode(idx, col, npart, dims)
       mapped_col = col
       ! Swap quark ↔ antiquark color indices (pairs position i ↔ position n+i)
       do ii = 1, min(n, m)
          tmp = mapped_col(ii)
          mapped_col(ii) = mapped_col(n + ii)
          mapped_col(n + ii) = tmp
       end do
       ! Apply M_perm to each gluon slot
       do p = n + m + 1, npart
          g = col(p) + 1                     ! 1-based CW gluon index
          mapped_col(p) = M_perm(g) - 1       ! apply M, back to 0-based
       end do
       ! Encode mapped state
       idx2 = 0
       do ii = 1, npart
          idx2 = idx2 + mapped_col(ii) * stride(ii-1)
       end do
       psi_out(idx2+1) = psi_in(idx+1)
    end do
  end subroutine apply_M_full

end program color_singlet_reduced_projection
