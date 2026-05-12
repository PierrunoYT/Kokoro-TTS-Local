$files = @('models.py', 'gradio_interface.py')
foreach ($f in $files) {
    $lines = [System.IO.File]::ReadAllLines($f)
    $fixed = $lines | ForEach-Object { $_.TrimEnd() }
    [System.IO.File]::WriteAllLines($f, $fixed, [System.Text.UTF8Encoding]::new($false))
    Write-Host "Done: $f"
}
