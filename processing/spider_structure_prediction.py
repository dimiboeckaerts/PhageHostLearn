"""
2ndary structure prediction of RBPs with Spider3

author: @dimiboeckaerts
"""

rbpbase = pd.read_csv(data_dir+'/RBPbase_250621_klebsiella_pneumoniae.csv')
rbp_counts = []
for phage in rbpbase['phage_nr']:
    rbp_counts.append(list(rbpbase['phage_nr']).count(phage))
ax = sns.countplot(x = rbp_counts)

# write klebs rbp files & adjust file_list for SPIDER3-S
spider_dir = '/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/SPIDER3-S'
klebs_rbps = '/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/Klebsiella_RBP_data/sequences/klebsiella_RBPs.fasta'
file_list = open(spider_dir+'/file_list', 'w')
kpids = []
for record in SeqIO.parse(klebs_rbps, 'fasta'):
    identifier = record.id
    kpids.append(identifier)
    sequence = str(record.seq)
    
    # write fasta
    fasta = open(spider_dir+'/example_data/seq/klebs_rbp_'+identifier+'.fasta', 'w')
    fasta.write('>'+identifier+'\n'+sequence+'\n')
    fasta.close()
    
    # write file_list
    file_list.write(identifier+' '+'./example_data/seq/klebs_rbp_'+identifier+'.fasta'+'\n')
file_list.close()

# compute KP32 secondary structures
spider_dir = '/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/SPIDER3-S'
command = 'cd '+spider_dir+ '; ./impute_script_np.sh'
ssprocess = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
ssout, sserr = ssprocess.communicate()
print(ssout)

def moving_average_SS(sequence, window_size=33):
    mas = []
    window_bit = int((window_size-1)/2)

    # compute margins outside of the window
    for i, character in enumerate(['H', 'E', 'C']):
        first_slice = sequence[:window_size]
        moving_average = np.zeros((len(sequence)))
        moving_average[:window_bit] = first_slice.count(character) / window_size
        last_slice = sequence[len(sequence)-window_size:]
        moving_average[len(sequence)-window_bit:] = last_slice.count(character) / window_size

        # sliding window
        for i in range(int(window_bit), int(len(sequence)-window_bit)):
            slice = sequence[int(i-window_bit):int(i+window_bit+1)]
            moving_average[i] = slice.count(character) / window_size
        mas.append(moving_average)
            
    return mas[0], mas[1], mas[2]

KP32s = ['ALT58497.1', 'ALT58498.1', 'YP_009198668.1', 'YP_009198669.1', 'APZ82804.1', 'APZ82805.1', 'APZ82847.1', 'APZ82848.1', 
'YP_002003830.1', 'YP_002003831.1', 'AWN07083.1', 'AWN07084.1', 'AWN07125.1', 'AWN07126.1', 'AWN07172.1', 'AWN07213.1', 'AWN07214.1', 
'YP_003347555.1', 'YP_003347556.1', 'YP_009215498.1', 'AOT28172.1', 'AOT28173.1', 'AOZ65569.1', 'AUV61507.1']
ma_B = []
ma_H = []
ma_C = []

for i, kp32 in enumerate(KP32s):
    results = pd.read_csv('/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/SPIDER3-S/example_data/outputs/'+kp32+'.i1', delim_whitespace=True, header=0)
    sstructs = ''.join([x for x in results['SS']])
    maHc, maBc, maCc = moving_average_SS(sstructs, window_size=50)
    ma_H.append(maHc)
    ma_B.append(maBc)
    ma_C.append(maCc)

index = 4
fig, ax = plt.subplots(figsize=(10,5))
ax.set_xlabel('position in the sequence')
ax.set_ylabel('Moving average (50 AA) B-sheet percentage')
ax.plot(ma_B[index], c='red', label='sheet')
ax.plot(ma_H[index], c='blue', label='helix')
ax.plot(ma_C[index], color='black', label='coil')
ax.legend()
fig.savefig('/Users/Dimi/Desktop/'+KP32s[index]+'.png', dpi=300)