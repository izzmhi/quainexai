<#
.SYNOPSIS
    Prove that speech recognition actually works on this machine.

.DESCRIPTION
    The voice test suite is entirely fakes - deliberately, so that `pytest` never
    opens a microphone or downloads a model. The cost of that choice is that
    nothing in CI proves faster-whisper can load a model and produce a
    transcript. This script closes that gap, and it is the only thing in the repo
    that does.

    It needs no microphone and no recording of a human voice: Windows' own speech
    synthesiser writes a WAV of a known phrase, and that phrase is what the
    transcript is checked against. Synthetic speech is a *harder* test than a
    clean human recording, not an easier one, so a pass here is meaningful.

    Run it after `pip install -e ".[voice]"`, and again whenever the Whisper
    model or its version changes.

    NOTE: this file is deliberately ASCII-only. Windows PowerShell 5.1 reads a
    BOM-less script as ANSI, so a single em-dash in a comment corrupts the parse
    and produces errors that point at unrelated lines.

.PARAMETER Phrase
    What to synthesise and expect back. The default includes the wake word so the
    wake-word path is exercised too.

.PARAMETER BaseUrl
    Where Quainex is listening.

.EXAMPLE
    .\scripts\verify_whisper.ps1
#>
[CmdletBinding()]
param(
    [string]$Phrase = "Quainex, take a screenshot",
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"

$wav = Join-Path $env:TEMP "quainex-whisper-check.wav"

Write-Host "1. Synthesising: $Phrase" -ForegroundColor Cyan
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    # Slowed slightly. Default SAPI pacing is clipped enough that Whisper
    # sometimes drops the first syllable, which would be a fault in the test
    # rather than in the recognition.
    $synth.Rate = -2
    $synth.SetOutputToWaveFile($wav)
    $synth.Speak($Phrase)
} finally {
    $synth.Dispose()
}

$size = (Get-Item $wav).Length
Write-Host "   wrote $wav ($([math]::Round($size / 1KB)) KB)" -ForegroundColor DarkGray
if ($size -lt 8000) {
    throw "The synthesised WAV is too small to contain speech. Check the SAPI voice."
}

Write-Host "2. Transcribing through $BaseUrl/voice/transcribe" -ForegroundColor Cyan
Write-Host "   (the first run downloads the Whisper model, which can take minutes)" -ForegroundColor DarkGray

$started = Get-Date
# curl.exe rather than Invoke-RestMethod: PowerShell 5.1's multipart support
# mangles binary bodies, which shows up as a corrupt-audio error rather than as
# the transport bug it is.
$upload = "audio=@$wav;type=audio/wav"
$response = & curl.exe -s -X POST "$BaseUrl/voice/transcribe" -F $upload
$elapsed = ((Get-Date) - $started).TotalSeconds

if (-not $response) { throw "No response from $BaseUrl. Is Quainex running?" }

$result = $response | ConvertFrom-Json
if ($result.error) {
    Write-Host "   FAILED: $($result.error.message)" -ForegroundColor Red
    exit 1
}

$text = $result.text
Write-Host "3. Transcript: $text" -ForegroundColor Green
$rt = [math]::Round($elapsed, 1)
Write-Host "   language=$($result.language) duration=$($result.duration_seconds)s round-trip=$rt s" -ForegroundColor DarkGray

# Word-level rather than exact-match: recognition legitimately differs on
# punctuation and casing, and demanding a byte-identical string would make this
# fail for reasons that have nothing to do with whether it works.
$expected = @(($Phrase -replace '[^\w\s]', '').ToLower().Split() | Where-Object { $_ })
$actual = ($text -replace '[^\w\s]', '').ToLower()
$hits = @($expected | Where-Object { $actual -match "\b$([regex]::Escape($_))" })
$score = if ($expected.Count) { $hits.Count / $expected.Count } else { 0 }
$pct = [math]::Round($score * 100)

Write-Host "4. Matched $($hits.Count)/$($expected.Count) expected words ($pct percent)" -ForegroundColor Cyan

Remove-Item $wav -ErrorAction SilentlyContinue

# 60 percent, not 100. The wake word is an invented proper noun that speech
# recognition mangles by design - that is why Quainex matches it phonetically
# rather than exactly. Demanding a perfect transcript here would be testing the
# synthesiser's diction, not Whisper's.
if ($score -lt 0.6) {
    Write-Host "FAIL: transcript does not resemble the phrase." -ForegroundColor Red
    exit 1
}

Write-Host "PASS: speech recognition works end to end on this machine." -ForegroundColor Green
