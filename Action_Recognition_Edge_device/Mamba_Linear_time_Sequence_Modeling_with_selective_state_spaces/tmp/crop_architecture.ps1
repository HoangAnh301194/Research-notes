Add-Type -AssemblyName System.Drawing

function Crop-Png {
  param(
    [string]$Source,
    [string]$Destination,
    [int]$X,
    [int]$Y,
    [int]$Width,
    [int]$Height
  )

  $sourcePath = (Resolve-Path -LiteralPath $Source).Path
  $destinationPath = Join-Path (Get-Location) $Destination
  $image = [System.Drawing.Bitmap]::FromFile($sourcePath)

  try {
    $rectangle = [System.Drawing.Rectangle]::new($X, $Y, $Width, $Height)
    $crop = $image.Clone($rectangle, $image.PixelFormat)
    try {
      $crop.Save($destinationPath, [System.Drawing.Imaging.ImageFormat]::Png)
    }
    finally {
      $crop.Dispose()
    }
  }
  finally {
    $image.Dispose()
  }
}

Crop-Png -Source 'tmp\online_architecture\h3-page-02.png' `
  -Destination 'assets\h3_architecture_crop.png' `
  -X 165 -Y 170 -Width 177 -Height 395

Crop-Png -Source 'tmp\online_architecture\transformer-page-03.png' `
  -Destination 'assets\transformer_architecture_crop.png' `
  -X 475 -Y 165 -Width 580 -Height 755

Crop-Png -Source 'tmp\principle_diagrams\s4-page-02.png' `
  -Destination 'assets\s4_lti_principle_crop.png' `
  -X 170 -Y 145 -Width 305 -Height 410

Crop-Png -Source 'tmp\principle_diagrams\transformer-page-04.png' `
  -Destination 'assets\self_attention_principle_crop.png' `
  -X 360 -Y 140 -Width 325 -Height 455

Crop-Png -Source 'tmp\principle_diagrams\shortformer-page-05.png' `
  -Destination 'assets\self_attention_tokens_crop.png' `
  -X 380 -Y 125 -Width 300 -Height 220

Crop-Png -Source 'assets\figure_2_synthetic_tasks.png' `
  -Destination 'assets\figure_2_synthetic_tasks_wide.png' `
  -X 0 -Y 0 -Width 1560 -Height 425

Crop-Png -Source 'tmp\alternative_ssm\lssl-page-02.png' `
  -Destination 'assets\lssl_recurrent_principle_crop.png' `
  -X 490 -Y 165 -Width 541 -Height 500
