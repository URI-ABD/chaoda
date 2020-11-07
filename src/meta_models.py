import numpy as np


def from_lr_cc_amean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[-0.02411, -0.41477, -0.13635, -0.34975, 0.72071, 0.42313],
        dtype=float,
    ), ratios))


def from_dt_cc_amean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius_ema <= 0.40425:
        if lfd <= 0.46660:
            if cardinality <= 0.40265:
                return 0.59595
            else:
                return 0.74640
        else:
            if radius <= 0.42805:
                return 0.57593
            else:
                return 0.31489
    else:
        if cardinality_ema <= 0.98520:
            if lfd_ema <= 0.52180:
                return 0.71740
            else:
                return 0.79801
        else:
            if radius_ema <= 0.95640:
                return 0.74108
            else:
                return 0.56305


def from_lr_sc_amean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[0.33306, -0.02295, -0.04441, 0.12345, -0.34289, -0.02357],
        dtype=float,
    ), ratios))


def from_dt_sc_amean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius_ema <= 0.40055:
        if cardinality <= 0.40675:
            if cardinality <= 0.38745:
                return 0.51003
            else:
                return 0.58825
        else:
            if lfd_ema <= 0.44845:
                return 0.70289
            else:
                return 0.79603
    else:
        if radius_ema <= 0.59970:
            if radius <= 0.54620:
                return 0.54361
            else:
                return 0.70228
        else:
            if cardinality_ema <= 0.41730:
                return 0.62450
            else:
                return 0.50195


def from_lr_gn_amean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[0.114, -0.01517, -0.0353, 0.11361, -0.1515, -0.0149],
        dtype=float,
    ), ratios))


def from_dt_gn_amean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius <= 0.33465:
        if lfd <= 0.47100:
            if cardinality_ema <= 0.33020:
                return 0.76740
            else:
                return 0.64800
        else:
            if radius <= 0.32900:
                return 0.51428
            else:
                return 0.71100
    else:
        if cardinality_ema <= 0.61940:
            if lfd <= 0.64135:
                return 0.48118
            else:
                return 0.15110
        else:
            if cardinality_ema <= 0.62335:
                return 0.54397
            else:
                return 0.49993


def from_lr_pc_amean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[-0.9154, -0.21377, 0.00784, -0.20117, 0.38132, 0.30135],
        dtype=float,
    ), ratios))


def from_dt_pc_amean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if cardinality_ema <= 0.98520:
        if lfd_ema <= 0.49985:
            if cardinality_ema <= 0.28060:
                return 0.63091
            else:
                return 0.74917
        else:
            if radius <= 0.46215:
                return 0.90113
            else:
                return 0.78494
    else:
        if radius_ema <= 0.95640:
            if radius <= 0.53455:
                return 0.36980
            else:
                return 0.75877
        else:
            if cardinality <= 0.49965:
                return 0.43831
            else:
                return 0.60677


def from_lr_rw_amean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[-1.19808, 0.09292, 0.15953, -0.13099, -0.09998, -0.05711],
        dtype=float,
    ), ratios))


def from_dt_rw_amean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if cardinality <= 0.49865:
        if radius <= 0.36650:
            if radius <= 0.22655:
                return 0.54412
            else:
                return 0.78440
        else:
            if radius_ema <= 0.99205:
                return 0.58541
            else:
                return 0.09765
    else:
        if radius_ema <= 0.82485:
            if radius_ema <= 0.69200:
                return 0.32483
            else:
                return 0.54499
        else:
            if radius_ema <= 0.84715:
                return 0.16433
            else:
                return 0.46028


def from_lr_sp_amean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[-1.08754, 0.03037, 0.25954, -0.07025, -0.18779, -0.09484],
        dtype=float,
    ), ratios))


def from_dt_sp_amean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if cardinality_ema <= 0.63045:
        if cardinality <= 0.43100:
            if radius_ema <= 0.48545:
                return 0.72291
            else:
                return 0.96796
        else:
            if lfd_ema <= 0.57300:
                return 0.58570
            else:
                return 0.43702
    else:
        if radius <= 0.35925:
            if cardinality_ema <= 0.98390:
                return 0.91820
            else:
                return 0.50000
        else:
            if radius <= 0.38885:
                return 0.01935
            else:
                return 0.46239


def from_lr_cc_gmean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[-0.1609, -0.34462, -0.13175, -0.16452, 0.45154, 0.33633],
        dtype=float,
    ), ratios))


def from_dt_cc_gmean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius_ema <= 0.19845:
        if cardinality_ema <= 0.25895:
            if radius_ema <= 0.17805:
                return 0.50618
            else:
                return 0.57986
        else:
            if cardinality <= 0.31070:
                return 0.82355
            else:
                return 0.85370
    else:
        if cardinality <= 0.44385:
            if radius_ema <= 0.31115:
                return 0.66483
            else:
                return 0.73878
        else:
            if radius_ema <= 0.99410:
                return 0.50662
            else:
                return 0.28395


def from_lr_sc_gmean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[0.02037, 0.00833, -0.05242, 0.08693, -0.23138, -0.02307],
        dtype=float,
    ), ratios))


def from_dt_sc_gmean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius_ema <= 0.46180:
        if lfd <= 0.46795:
            if radius <= 0.29385:
                return 0.71186
            else:
                return 0.59252
        else:
            if lfd <= 0.47680:
                return 0.52131
            else:
                return 0.50068
    else:
        if lfd <= 0.31065:
            if radius_ema <= 0.64630:
                return 0.75000
            else:
                return 0.50063
        else:
            if cardinality_ema <= 0.24920:
                return 0.62450
            else:
                return 0.50068


def from_lr_gn_gmean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[0.00067, 0.02239, -0.00858, 0.05142, -0.07763, -0.0288],
        dtype=float,
    ), ratios))


def from_dt_gn_gmean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if cardinality_ema <= 0.22530:
        return 0.80650
    else:
        if radius <= 0.13255:
            if radius <= 0.12390:
                return 0.43480
            else:
                return 0.12870
        else:
            if radius_ema <= 0.26600:
                return 0.55391
            else:
                return 0.49586


def from_lr_pc_gmean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[-0.26459, -0.28155, -0.01405, -0.02068, 0.11857, 0.2612],
        dtype=float,
    ), ratios))


def from_dt_pc_gmean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if cardinality <= 0.44385:
        if radius_ema <= 0.90440:
            if cardinality <= 0.25905:
                return 0.87253
            else:
                return 0.75220
        else:
            if radius <= 0.87590:
                return 0.70677
            else:
                return 0.58471
    else:
        if radius_ema <= 0.99410:
            if cardinality <= 0.48535:
                return 0.68455
            else:
                return 0.44731
        else:
            if cardinality_ema <= 0.99995:
                return 0.23080
            else:
                return 0.29380


def from_lr_rw_gmean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[-0.06811, 0.01474, 0.12457, -0.02628, -0.26829, -0.06079],
        dtype=float,
    ), ratios))


def from_dt_rw_gmean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if cardinality_ema <= 0.63195:
        if radius <= 0.25805:
            if cardinality <= 0.17885:
                return 0.93753
            else:
                return 0.67074
        else:
            if radius_ema <= 0.39650:
                return 0.66028
            else:
                return 0.53630
    else:
        if cardinality_ema <= 0.83730:
            if radius <= 0.34920:
                return 0.20814
            else:
                return 0.41664
        else:
            if radius <= 0.83135:
                return 0.46577
            else:
                return 0.53153


def from_lr_sp_gmean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[-0.11242, -0.00448, 0.25717, -0.00151, -0.32649, -0.11889],
        dtype=float,
    ), ratios))


def from_dt_sp_gmean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius <= 0.29470:
        if lfd <= 0.59230:
            if cardinality_ema <= 0.60325:
                return 0.68508
            else:
                return 0.44042
        else:
            if radius <= 0.09490:
                return 0.60980
            else:
                return 0.95561
    else:
        if radius_ema <= 0.46050:
            if cardinality <= 0.30365:
                return 0.51666
            else:
                return 0.74438
        else:
            if radius <= 0.31415:
                return 0.01437
            else:
                return 0.47655


def from_lr_cc_hmean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[-0.12509, -0.22071, 0.02034, -0.06233, 0.22682, 0.19787],
        dtype=float,
    ), ratios))


def from_dt_cc_hmean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius_ema <= 0.00005:
        if radius <= 0.01200:
            if lfd <= 0.41120:
                return 0.62006
            else:
                return 0.49744
        else:
            if cardinality_ema <= 0.14790:
                return 0.24480
            else:
                return 0.06393
    else:
        if cardinality <= 0.37945:
            if lfd_ema <= 0.35060:
                return 0.67299
            else:
                return 0.74417
        else:
            if radius_ema <= 0.99410:
                return 0.50662
            else:
                return 0.28395


def from_lr_sc_hmean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[-0.02959, 0.02071, -0.07717, 0.0336, -0.11739, -0.01307],
        dtype=float,
    ), ratios))


def from_dt_sc_hmean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius_ema <= 0.35810:
        if radius <= 0.18130:
            if radius <= 0.09925:
                return 0.50084
            else:
                return 0.55036
        else:
            if radius <= 0.21295:
                return 0.74399
            else:
                return 0.60160
    else:
        if radius_ema <= 0.48595:
            if radius_ema <= 0.48040:
                return 0.51980
            else:
                return 0.83330
        else:
            if radius <= 0.00095:
                return 0.50966
            else:
                return 0.50030


def from_lr_gn_hmean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[-0.02357, 0.03249, -0.00445, 0.01641, -0.03078, -0.02866],
        dtype=float,
    ), ratios))


def from_dt_gn_hmean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius <= 0.16545:
        if lfd <= 0.55780:
            if radius <= 0.14295:
                return 0.47722
            else:
                return 0.35550
        else:
            if lfd_ema <= 0.43410:
                return 0.26740
            else:
                return 0.46822
    else:
        if radius <= 0.26470:
            if radius <= 0.26200:
                return 0.54189
            else:
                return 0.93660
        else:
            if radius_ema <= 0.24820:
                return 0.15130
            else:
                return 0.49678


def from_lr_pc_hmean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[-0.19524, -0.23807, 0.08194, 0.04003, 0.04819, 0.19191],
        dtype=float,
    ), ratios))


def from_dt_pc_hmean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius <= 0.16655:
        if radius <= 0.00035:
            if radius <= 0.00005:
                return 0.56533
            else:
                return 0.68870
        else:
            if radius <= 0.00135:
                return 0.79880
            else:
                return 0.89754
    else:
        if cardinality <= 0.37945:
            if lfd_ema <= 0.32355:
                return 0.67852
            else:
                return 0.74700
        else:
            if radius_ema <= 0.99410:
                return 0.50662
            else:
                return 0.27805


def from_lr_rw_hmean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[0.02713, 0.01326, 0.08829, -0.02638, -0.19995, -0.03986],
        dtype=float,
    ), ratios))


def from_dt_rw_hmean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius_ema <= 0.36595:
        if lfd <= 0.51600:
            if cardinality <= 0.04070:
                return 0.07160
            else:
                return 0.62851
        else:
            if cardinality_ema <= 0.24855:
                return 0.81270
            else:
                return 0.95925
    else:
        if radius <= 0.00035:
            if radius <= 0.00025:
                return 0.97660
            else:
                return 0.94420
        else:
            if cardinality_ema <= 0.15315:
                return 0.66206
            else:
                return 0.47298


def from_lr_sp_hmean(ratios: np.array) -> float:
    return float(np.dot(np.asarray(
        a=[-0.02136, -0.01068, 0.2275, 0.00078, -0.24275, -0.08647],
        dtype=float,
    ), ratios))


def from_dt_sp_hmean(ratios: np.array) -> float:
    cardinality, radius, lfd, cardinality_ema, radius_ema, lfd_ema = tuple(ratios)
    if radius_ema <= 0.36595:
        if lfd <= 0.51600:
            if cardinality <= 0.04070:
                return 0.01970
            else:
                return 0.64370
        else:
            if cardinality_ema <= 0.24855:
                return 0.81225
            else:
                return 0.96880
    else:
        if radius <= 0.00175:
            if cardinality <= 0.09040:
                return 0.93782
            else:
                return 0.30925
        else:
            if lfd_ema <= 0.41035:
                return 0.49243
            else:
                return 0.41559


META_MODELS = {
    'from_lr_cc_amean': from_lr_cc_amean,
    'from_dt_cc_amean': from_dt_cc_amean,
    'from_lr_sc_amean': from_lr_sc_amean,
    'from_dt_sc_amean': from_dt_sc_amean,
    'from_lr_gn_amean': from_lr_gn_amean,
    'from_dt_gn_amean': from_dt_gn_amean,
    'from_lr_pc_amean': from_lr_pc_amean,
    'from_dt_pc_amean': from_dt_pc_amean,
    'from_lr_rw_amean': from_lr_rw_amean,
    'from_dt_rw_amean': from_dt_rw_amean,
    'from_lr_sp_amean': from_lr_sp_amean,
    'from_dt_sp_amean': from_dt_sp_amean,
    'from_lr_cc_gmean': from_lr_cc_gmean,
    'from_dt_cc_gmean': from_dt_cc_gmean,
    'from_lr_sc_gmean': from_lr_sc_gmean,
    'from_dt_sc_gmean': from_dt_sc_gmean,
    'from_lr_gn_gmean': from_lr_gn_gmean,
    'from_dt_gn_gmean': from_dt_gn_gmean,
    'from_lr_pc_gmean': from_lr_pc_gmean,
    'from_dt_pc_gmean': from_dt_pc_gmean,
    'from_lr_rw_gmean': from_lr_rw_gmean,
    'from_dt_rw_gmean': from_dt_rw_gmean,
    'from_lr_sp_gmean': from_lr_sp_gmean,
    'from_dt_sp_gmean': from_dt_sp_gmean,
    'from_lr_cc_hmean': from_lr_cc_hmean,
    'from_dt_cc_hmean': from_dt_cc_hmean,
    'from_lr_sc_hmean': from_lr_sc_hmean,
    'from_dt_sc_hmean': from_dt_sc_hmean,
    'from_lr_gn_hmean': from_lr_gn_hmean,
    'from_dt_gn_hmean': from_dt_gn_hmean,
    'from_lr_pc_hmean': from_lr_pc_hmean,
    'from_dt_pc_hmean': from_dt_pc_hmean,
    'from_lr_rw_hmean': from_lr_rw_hmean,
    'from_dt_rw_hmean': from_dt_rw_hmean,
    'from_lr_sp_hmean': from_lr_sp_hmean,
    'from_dt_sp_hmean': from_dt_sp_hmean,
}
