import sags
import swells
import interruptions
import flickers
import harmonics
import spikes
import osc_transients
import sags_harmonics
import swells_harmonics
import interruptions_harmonics

# Define the number of samples to be created for each of the PQD signals
n = 1000

# Generate the signal samples
sags.generate_signals(n)
swells.generate_signals(n)
interruptions.generate_signals(n)
flickers.generate_signals(n)
harmonics.generate_signals(n)
spikes.generate_signals(n)
osc_transients.generate_signals(n)
sags_harmonics.generate_signals(n)
swells_harmonics.generate_signals(n)
interruptions_harmonics.generate_signals(n)