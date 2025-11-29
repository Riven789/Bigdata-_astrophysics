from ml4gw.waveforms import IMRPhenomD
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
from ml4gw.types import BatchTensor
from ml4gw import distributions
from ml4gw.waveforms.conversion import chirp_mass_and_mass_ratio_to_components
from ml4gw import gw
from ml4gw.distributions import PowerLaw, Sine, Cosine, DeltaFunction
from torch.distributions import Uniform
from ml4gw.gw import get_ifo_geometry, compute_observed_strain
from ml4gw.gw import compute_ifo_snr, compute_network_snr


import torch
from torch.distributions import Uniform


def hphc(num, approx=IMRPhenomD(), sample_rate=2048, duration=30, f_min=20, f_max=200, right_pas=15, ifos=['H1', 'L1']):

    waveform = TimeDomainCBCWaveformGenerator(approx, sample_rate, duration, f_min, f_max, right_pas)
    
    chirp_mass=Uniform(
            torch.as_tensor(10, dtype=torch.float32),
            torch.as_tensor(100, dtype=torch.float32),
            validate_args=False,
        ).sample((num, ))
    mass_ratio=Uniform(
            torch.as_tensor(0.125, dtype=torch.float32),
            torch.as_tensor(0.999, dtype=torch.float32),
            validate_args=False,
        ).sample((num, ))
    distance=Uniform(
            torch.as_tensor(1000, dtype=torch.float32),
            torch.as_tensor(3100, dtype=torch.float32),
            validate_args=False,
        ).sample((num, ))
    inclination=distributions.Sine(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(torch.pi, dtype=torch.float32),
            validate_args=False,
        ).sample((num, ))
    phic=Uniform(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(2 * torch.pi, dtype=torch.float32),
            validate_args=False,
        ).sample((num, ))
    chi1=Uniform(
            torch.as_tensor(-0.999, dtype=torch.float32),
            torch.as_tensor(0.999, dtype=torch.float32),
            validate_args=False,
        ).sample((num, ))
    chi2=Uniform(
            torch.as_tensor(-0.999, dtype=torch.float32),
            torch.as_tensor(0.999, dtype=torch.float32),
            validate_args=False,
        ).sample((num, ))

    parameters = {'chirp_mass':chirp_mass, 'mass_ratio':mass_ratio, 'distance':distance,
              'inclination':inclination, 'phic':phic, 'chi1':chi1, 'chi2':chi2}


    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(chirp_mass, mass_ratio)
    parameters['mass_1'], parameters['mass_2'] = mass_1, mass_2
    parameters['s1z'], parameters['s2z'] = parameters["chi1"], parameters["chi2"]

    #return waveform(**parameters), parameters

    plus, cross = waveform(**parameters)
    plus = plus.to(dtype=torch.float32)
    cross = cross.to(dtype=torch.float32)

    dec=distributions.Sine(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(torch.pi, dtype=torch.float32),
            validate_args=False,
        ).sample((num, ))
    ra=distributions.Sine(
            torch.as_tensor(0, dtype=torch.float32),
            torch.as_tensor(torch.pi, dtype=torch.float32),
            validate_args=False,
        ).sample((num, ))

    responses = (plus, cross)
    parameters['ra'] = ra
    parameters['dec'] = dec
    parameters.pop('mass_1')
    parameters.pop('mass_2')
    parameters.pop('s1z')
    parameters.pop('s2z')
    
    return responses, 


param_dict = {
    "chirp_mass": PowerLaw(10, 100, -2.35),
    "mass_ratio": Uniform(0.125, 0.999),
    "chi1": Uniform(-0.999, 0.999),
    "chi2": Uniform(-0.999, 0.999),
    "distance": PowerLaw(100, 500, 2),
    "phic": DeltaFunction(0),
    "inclination": Sine(),
}

dec = Cosine()
psi = Uniform(0, torch.pi)
phi = Uniform(-torch.pi, torch.pi)

device = 'cpu'

def new_hphc(num, approx=IMRPhenomD(), duration=8, sample_rate=2048, f_min = 20, f_max = 1024, f_ref = 20, psd=None):

    nyquist = sample_rate / 2
    num_samples = int(duration * sample_rate)
    num_freqs = num_samples // 2 + 1

    # Create an array of frequency values at which to generate our waveform
    # At the moment, only frequency-domain approximants have been implemented
    frequencies = torch.linspace(0, nyquist, num_freqs).to(device)
    freq_mask = (frequencies >= f_min) * (frequencies < f_max).to(device)

    params = {
    k: v.sample((num,)).to(device) for k, v in param_dict.items()
    }

    hc_f, hp_f = approx(f=frequencies[freq_mask], f_ref=f_ref, **params)

    shape = (hc_f.shape[0], num_freqs)
    hc_spectrum = torch.zeros(shape, dtype=hc_f.dtype, device=device)
    hp_spectrum = torch.zeros(shape, dtype=hc_f.dtype, device=device)

    # fill the spectrum with the
    # hc and hp values at the specified frequencies
    hc_spectrum[:, freq_mask] = hc_f
    hp_spectrum[:, freq_mask] = hp_f

    # now, irfft and scale the waveforms by sample_rate
    hc, hp = torch.fft.irfft(hc_spectrum), torch.fft.irfft(hp_spectrum)
    hc *= sample_rate
    hp *= sample_rate

    # The coalescence point is placed at the right edge, so shift it to
    # give some room for ringdown
    ringdown_duration = 3
    ringdown_size = int(ringdown_duration * sample_rate)
    hc = torch.roll(hc, -ringdown_size, dims=-1)
    hp = torch.roll(hp, -ringdown_size, dims=-1)


    ## projections
    ifos = ["H1", "L1"]
    tensors, vertices = get_ifo_geometry(*ifos)
    dec_samples = dec.sample((num,)).to(device)
    psi_samples = psi.sample((num,)).to(device)
    phi_samples = phi.sample((num,)).to(device)

    # Pass the detector geometry, along with the polarizations and sky parameters,
    # to get the observed strain
    waveforms = compute_observed_strain(
        dec=dec_samples,
        psi=psi_samples,
        phi=phi_samples,
        detector_tensors=tensors.to(device),
        detector_vertices=vertices.to(device),
        sample_rate=sample_rate,
        cross=hc,
        plus=hp,
    )

    params['dec'] = dec_samples
    params['psi'] = psi_samples
    params['phi'] = phi_samples

    if psd is None:
        return waveforms, params

    else:

        freqs = torch.linspace(0, nyquist, psd.shape[-1])

        if psd.shape[-1] != num_freqs:
            # Adding dummy dimensions for consistency
            while psd.ndim < 3:
                psd = psd[None]
            psd = torch.nn.functional.interpolate(
                psd, size=(num_freqs,), mode="linear"
            )

        # We can compute both the individual and network SNRs
        # The SNR calculation starts at the minimum frequency we
        # specified earlier and goes to the maximum
        # TODO: There's probably no reason to have multiple functions
        
        snr = compute_network_snr(
            responses=waveforms, psd=psd, sample_rate=sample_rate, highpass=f_min
        )

        h1_snr = compute_ifo_snr(responses=waveforms[:, 0], psd=psd[:, 0],
                                 sample_rate=sample_rate,highpass=f_min)
        
        l1_snr = compute_ifo_snr(responses=waveforms[:, 1], psd=psd[:, 1],
                                 sample_rate=sample_rate,highpass=f_min)

        params['snr'] = snr

        return waveforms, params, h1_snr, l1_snr



    
    

    
