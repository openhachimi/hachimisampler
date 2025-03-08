using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

class Program
{
    static readonly HttpClient client = new HttpClient();

    static async Task Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("No arguments provided.");
            Environment.Exit(1);
        }

        try
        {
            var postFields = string.Join(" ", args);
            System.Console.WriteLine($"Sending: {postFields}");
            var content = new StringContent(postFields, Encoding.UTF8, "application/x-www-form-urlencoded");

            HttpResponseMessage response = await client.PostAsync("http://127.0.0.1:8572", content); 

            if (response.IsSuccessStatusCode)
            {
                Console.WriteLine("Success: Straycat Resampled Sucessfully");
            } 
            else if (response.StatusCode == System.Net.HttpStatusCode.BadRequest)
            {
                Console.Error.WriteLine("Error: StrayCat got an incorrect amount of arguments or the arguments were out of order. Please check the input data before continuing.");
            }
            else if (response.StatusCode == System.Net.HttpStatusCode.InternalServerError)
            {
                Console.Error.WriteLine($"Error: An Internal Error occured in StraycatServer. Check your voicebank wav files to ensure they are the correct format. More details:\n{await response.Content.ReadAsStringAsync()}");
            }
            else
            {
                Console.Error.WriteLine($"Error: Straycat returned {response.StatusCode}");
            }
        }
        catch (HttpRequestException e)
        {
            Console.Error.WriteLine($"Request exception: {e.Message}\nIs straycat_server running?");
        }
    }
}