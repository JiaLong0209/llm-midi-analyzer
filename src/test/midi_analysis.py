import music21

def analyze_midi_deeply(file_path: str):
    print(f"🎵 深入分析 MIDI 檔案: {file_path}")
    print("=" * 50)
    
    # 1. 載入 MIDI 檔案轉換為 music21 的 Score (總譜) 物件
    # music21 是一個「樹狀結構 (Tree Structure)」的系統，類似網頁的 DOM (HTML)。
    # Score 包含 Part (聲部/樂器)，Part 包含 Measure (小節)，Measure 包含 Note/Chord/Rest 等。
    try:
        score = music21.converter.parse(file_path)
    except Exception as e:
        print(f"無法讀取檔案: {e}")
        return

    # 2. 獲取全域詮釋資料 (Metadata)
    print("\n[全域樂理特徵]")
    
    # 為什麼要用 getElementsByClass? 
    # 因為一個 Stream (如 score) 裡面塞滿了各種不同的物件 (音符、休止符、速度記號、調號)。
    # getElementsByClass 就像是資料庫的 SELECT 語法，專門幫你把特定類型的物件「濾」出來。
    
    # 尋找調號 (Key Signature)
    # .flat 會把階層架構「壓平」，讓你不用自己寫迴圈一層一層找 (Score -> Part -> Measure -> Key)
    key_sigs = score.flat.getElementsByClass(music21.key.KeySignature)
    if key_sigs:
        print(f"👉 初始調性: {key_sigs[0].asKey().name} (共有 {len(key_sigs)} 個調號變更)")
    else:
        # 如果 MIDI 沒存調號，可以請 music21 演算法直接分析
        estimated_key = score.analyze('key')
        print(f"👉 演算法猜測調性: {estimated_key.name} (信心度: {estimated_key.correlationCoefficient:.2f})")

    # 尋找拍號 (Time Signature)
    time_sigs = score.flat.getElementsByClass(music21.meter.TimeSignature)
    if time_sigs:
        print(f"👉 初始拍號: {time_sigs[0].ratioString} (共有 {len(time_sigs)} 個拍號變更)")

    # 尋找速度 (Tempo)
    tempos = score.flat.getElementsByClass(music21.tempo.MetronomeMark)
    if tempos:
        print(f"👉 初始速度: {tempos[0].number} BPM")

    # 3. 遍歷所有的樂器/聲部 (Parts)
    print("\n[樂器聲部分析]")
    parts = score.getElementsByClass(music21.stream.Part)
    
    for i, part in enumerate(parts):
        # 嘗試取得樂器名稱
        inst = part.getInstrument()
        inst_name = inst.instrumentName if inst and inst.instrumentName else "未知樂器"
        print(f"🎺 聲部 {i+1}: {inst_name}")
        
        # 4. 深入到小節 (Measures)
        measures = part.getElementsByClass(music21.stream.Measure)
        print(f"   總小節數: {len(measures)}")
        
        # 我們只挑前兩個小節來印出詳細結構，不然終端機會被洗版
        for measure in measures:
            print(f"   ► 小節 {measure.number}:")
            
            # 從小節裡挑出「音符 (Note)」
            for note in measure.getElementsByClass(music21.note.Note):
                print(f"      🎵 音符: {note.nameWithOctave:4} | 位置: @{note.offset:<4} | 長度: {note.quarterLength} 拍 | 力度: {note.volume.velocity if note.volume.velocity else '預設'}")
                
            # 從小節裡挑出「和弦 (Chord)」(如果 MIDI 原本就是把音符寫在一起，就會被解析成 Chord)
            for chord in measure.getElementsByClass(music21.chord.Chord):
                pitches = ",".join(p.nameWithOctave for p in chord.pitches)
                print(f"      🎹 和弦: [{pitches:10}] | 位置: @{chord.offset:<4} | 俗稱: {chord.pitchedCommonName}")
                
            # 從小節裡挑出「休止符 (Rest)」
            for rest in measure.getElementsByClass(music21.note.Rest):
                print(f"      🔕 休止符 | 位置: @{rest.offset:<4} | 長度: {rest.quarterLength} 拍")

    # 5. 特殊功能：整首和弦化 (Chordify)
    # 這是 music21 超強的功能！它會把所有不同聲部、不同樂器的音，
    # 根據相同的發聲時間，垂直「拍扁」擠成一連串的實體和弦，這對分析總譜對位非常有用。
    print("\n[進階運算: 全曲垂直和弦化 (Chordify) 預覽前 3 個和弦]")
    chordified_score = score.chordify()
    first_few_chords = chordified_score.flat.getElementsByClass(music21.chord.Chord)
    for c in first_few_chords:
         print(f"📍 @{c.offset}: {c.pitchedCommonName} (組成音: {','.join(p.nameWithOctave for p in c.pitches)})")

if __name__ == "__main__":
    import sys
    # 預設讀取我們前面的測試檔
    test_file = "midi/no.12_v2.mid"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    analyze_midi_deeply(test_file)
